"""Data loading, cleaning, selection, preprocessing, and Batch Helper logic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import re

import numpy as np
import pandas as pd


@dataclass
class ExcelLoadResult:
    """Processed Excel data returned by the application data loader."""

    dataframe: pd.DataFrame
    batch_source_df: pd.DataFrame
    all_devices: list[str]
    analysis_devices: list[str]

def data_cleaning(df):
    """
    Clean and prepare the raw dataframe.
    """

    df = df.copy()

    df["Data"] = pd.to_datetime(
        df["Data"],
        dayfirst=True,
        errors="coerce"
    )

    df["Year"] = df["Data"].dt.year
    df["Month"] = df["Data"].dt.month
    df["Week"] = df["Data"].dt.isocalendar().week
    df["Day"] = df["Data"].dt.day

    return df

def add_effective_unit(df, device_config=None):
    """
    Recreate the notebook's effective-unit calculation before the TA-only
    filter is applied.

    effective_unit = total_unit - dummy_unit
    dummy_unit counts MachineRejectCode values 99, 6, and 206.
    """
    out = df.copy()

    if "effective_unit" in out.columns:
        return out

    configs = {
        "SOLARIS-XRF3": (17, 76),
        "VEGA": (15, 64),
        "SQUID-ICD82PRV": (17, 72),
    }

    if isinstance(device_config, dict):
        for device, cfg in device_config.items():
            if isinstance(cfg, dict):
                try:
                    configs[str(device)] = (int(cfg["rows"]), int(cfg["cols"]))
                except (KeyError, TypeError, ValueError):
                    pass

    invalid_codes = {99, 6, 206}

    if "USMDevice" not in out.columns:
        raise ValueError("USMDevice is required to calculate effective_unit.")
    if "MESLotID" not in out.columns or "StripID" not in out.columns:
        raise ValueError("MESLotID and StripID are required to calculate effective_unit.")
    if "MachineRejectCode" not in out.columns:
        raise ValueError("MachineRejectCode is required to calculate effective_unit.")

    out["_rows"] = out["USMDevice"].map(lambda x: configs.get(str(x), (0, 0))[0])
    out["_cols"] = out["USMDevice"].map(lambda x: configs.get(str(x), (0, 0))[1])
    out["_total_unit"] = out["_rows"] * out["_cols"]
    out["_dummy_unit"] = out["MachineRejectCode"].apply(
        lambda x: 1 if x in invalid_codes else 0
    )

    strip_units = (
        out.groupby(["MESLotID", "StripID"], dropna=False)
        .agg(
            effective_unit=("_total_unit", "first"),
            dummy_unit=("_dummy_unit", "sum"),
            total_unit=("_total_unit", "first"),
        )
        .reset_index()
    )
    strip_units["effective_unit"] = (
        strip_units["total_unit"] - strip_units["dummy_unit"]
    ).clip(lower=1)

    out = out.merge(
        strip_units[["MESLotID", "StripID", "effective_unit"]],
        on=["MESLotID", "StripID"],
        how="left",
    )

    return out.drop(columns=["_rows", "_cols", "_total_unit", "_dummy_unit"], errors="ignore")

def add_strip_anomaly_feature(
    df,
    z_threshold=3.0,
    device_config=None,
    criteria_map=None,
):
    """
    Calculate strip-level anomaly using Median + MAD, separately by Criteria.

    The dataframe passed to this function is expected to already contain only
    the user's selected analysis population (selected device(s) plus the
    selected date range / batch(es)).

    Devices sharing the same Criteria are pooled together. Criteria A and B
    are never mixed when calculating the Median/MAD baseline.
    """
    if df is None or df.empty:
        return df.copy()

    work = df.copy()

    if criteria_map is None:
        raise ValueError("criteria_map must be provided for Criteria-based anomaly filtering.")

    work["_criteria"] = work["USMDevice"].astype(str).map(criteria_map)
    missing_criteria = work["_criteria"].isna() | work["_criteria"].eq("")
    if missing_criteria.any():
        missing_devices = sorted(
            work.loc[missing_criteria, "USMDevice"].astype(str).unique()
        )
        raise ValueError(
            "No valid Criteria A/B configuration found for: "
            + ", ".join(missing_devices)
        )

    work["_reject"] = (
        work["UserRejectCode"].astype(str).str.strip().eq("TA")
    ).astype(int)

    # One row per strip.  Device is kept here only to uniquely identify the
    # strip when merging the calculated result back.  It is NOT used to
    # calculate the Criteria baseline.
    strip_df = (
        work.groupby(
            ["USMDevice", "_criteria", "MESLotID", "StripID"],
            dropna=False,
        )
        .agg(
            reject_count=("_reject", "sum"),
            effective_unit=("effective_unit", "first"),
        )
        .reset_index()
    )

    strip_df["strip_level_reject_rate"] = (
        strip_df["reject_count"] / strip_df["effective_unit"]
    )

    # Calculate Median and MAD separately for each Criteria.
    stats = (
        strip_df.groupby("_criteria")["strip_level_reject_rate"]
        .agg(median="median")
        .reset_index()
    )

    mad_df = (
        strip_df.groupby("_criteria")["strip_level_reject_rate"]
        .apply(lambda x: np.median(np.abs(x - np.median(x))))
        .reset_index(name="mad")
    )

    stats = stats.merge(mad_df, on="_criteria", how="left")
    stats["sigma_mad"] = 1.4826 * stats["mad"]

    # Avoid division by zero when all strips in a Criteria group have the
    # same reject rate. In that case all Z-scores are set to zero below.
    strip_df = strip_df.merge(
        stats[["_criteria", "median", "mad", "sigma_mad"]],
        on="_criteria",
        how="left",
    )

    sigma_values = pd.to_numeric(strip_df["sigma_mad"], errors="coerce")
    valid = np.isfinite(sigma_values) & (sigma_values > 0)
    strip_df["strip_level_z"] = 0.0
    strip_df.loc[valid, "strip_level_z"] = (
        strip_df.loc[valid, "strip_level_reject_rate"]
        - strip_df.loc[valid, "median"]
    ) / sigma_values.loc[valid]

    strip_df["strip_ucl"] = (
        strip_df["median"] + z_threshold * strip_df["sigma_mad"]
    )

    strip_df["strip_level_anomaly"] = (
        strip_df["strip_level_z"].abs() > z_threshold
    )

    return work.merge(
        strip_df[
            [
                "USMDevice",
                "MESLotID",
                "StripID",
                "strip_ucl",
                "strip_level_reject_rate",
                "strip_level_z",
                "strip_level_anomaly",
            ]
        ],
        on=["USMDevice", "MESLotID", "StripID"],
        how="left",
    )


def load_excel_data(
    file_path: str,
    *,
    device_config: dict[str, dict[str, Any]],
    data_cleaning: Any,
    add_effective_unit: Any,
) -> ExcelLoadResult:
    """Read, clean, enrich, and split uploaded Excel data."""
    raw_df = pd.read_excel(file_path)
    dataframe = data_cleaning(raw_df)
    dataframe = add_effective_unit(
        dataframe,
        device_config=device_config,
    )
    dataframe = dataframe.dropna(
        subset=["Data", "USMDevice"],
    ).copy()

    batch_source_df = dataframe.copy()

    all_devices = sorted(
        dataframe["USMDevice"]
        .dropna()
        .astype(str)
        .str.strip()
        .loc[lambda values: values.ne("")]
        .unique()
        .tolist()
    )

    if "UserRejectCode" in dataframe.columns:
        dataframe = dataframe[
            dataframe["UserRejectCode"]
            .astype(str)
            .str.strip()
            .eq("TA")
        ].copy()

    analysis_devices = sorted(
        dataframe["USMDevice"]
        .dropna()
        .astype(str)
        .str.strip()
        .loc[lambda values: values.ne("")]
        .unique()
        .tolist()
    )

    return ExcelLoadResult(
        dataframe=dataframe,
        batch_source_df=batch_source_df,
        all_devices=all_devices,
        analysis_devices=analysis_devices,
    )


def filter_device_date_population(dataframe, devices, start_value, end_value):
    """Return selected devices inside the inclusive date range."""
    if dataframe is None or dataframe.empty:
        return pd.DataFrame() if dataframe is None else dataframe.iloc[0:0].copy()

    start = pd.to_datetime(start_value).normalize()
    end = pd.to_datetime(end_value).normalize()
    if start > end:
        raise ValueError("Start date cannot be later than end date.")

    selected = {str(device).strip() for device in devices}
    device_values = dataframe["USMDevice"].astype(str).str.strip()

    return dataframe[
        dataframe["Data"].notna()
        & device_values.isin(selected)
        & (dataframe["Data"] >= start)
        & (dataframe["Data"] < end + pd.Timedelta(days=1))
    ].copy()


def get_analysis_population(
    dataframe,
    devices,
    start_value,
    end_value,
    use_batches,
    batch_mode,
    selected_batch_lots,
):
    """Build the exact dataframe allowed into downstream analysis."""
    population = filter_device_date_population(
        dataframe,
        devices,
        start_value,
        end_value,
    )

    if use_batches and batch_mode == "Batch Helper":
        population = filter_by_selected_batches(population, selected_batch_lots)

    return population


def apply_anomaly_preprocessing(
    dataframe,
    enabled,
    z_threshold,
    device_config,
    add_strip_anomaly_feature,
):
    """Exclude anomalous strips from the supplied analysis population."""
    if dataframe is None or dataframe.empty:
        return dataframe

    if not enabled:
        return dataframe

    criteria_map = {
        str(device): str(config.get("criteria", "")).strip().upper()
        for device, config in device_config.items()
        if isinstance(config, dict)
    }

    result = add_strip_anomaly_feature(
        dataframe,
        z_threshold=z_threshold,
        device_config=device_config,
        criteria_map=criteria_map,
    )
    return result[~result["strip_level_anomaly"]].copy()


def prepare_device_analysis_data(
    dataframe,
    selected_devices,
    analysis_mode,
    start,
    end,
    n_days,
    device_data,
    get_strip_id_column,
):
    """Prepare per-device analysis data and optional aggregation groups."""
    strip_col = get_strip_id_column(dataframe)
    if strip_col is None:
        return None, []

    device_data.clear()
    devices_with_data = []
    device_values = dataframe["USMDevice"].astype(str)

    for device in selected_devices:
        device_df = dataframe[device_values == str(device)].copy()

        if device_df.empty:
            continue

        devices_with_data.append(device)

        if analysis_mode == "Individual Strip":
            device_data[device] = {
                "df": device_df,
                "strip_col": strip_col,
            }

        elif analysis_mode == "N-Day Aggregation":
            groups = {}
            current_start = start

            while current_start <= end:
                current_end = min(
                    current_start + pd.Timedelta(days=n_days - 1),
                    end,
                )
                group_df = device_df[
                    (device_df["Data"] >= current_start)
                    & (
                        device_df["Data"]
                        < current_end + pd.Timedelta(days=1)
                    )
                ].copy()

                if not group_df.empty:
                    label = (
                        f"{current_start.strftime('%Y-%m-%d')} to "
                        f"{current_end.strftime('%Y-%m-%d')}"
                    )
                    groups[label] = {
                        "df": group_df,
                        "number_of_strips": group_df[
                            strip_col
                        ].dropna().nunique(),
                        "label": label,
                    }

                current_start = current_end + pd.Timedelta(days=1)

            device_data[device] = {
                "df": device_df,
                "strip_col": strip_col,
                "groups": groups,
            }

        elif analysis_mode == "Weekly":
            device_df = device_df.copy()
            device_df["WeekStart"] = (
                device_df["Data"]
                .dt.to_period("W-SUN")
                .apply(lambda period: period.start_time)
            )

            groups = {}
            for week_start, group_df in device_df.groupby(
                "WeekStart",
                sort=True,
            ):
                week_start = pd.Timestamp(week_start)
                week_end = week_start + pd.Timedelta(days=6)
                display_start = max(week_start, start)
                display_end = min(week_end, end)
                label = (
                    f"{display_start.strftime('%Y-%m-%d')} to "
                    f"{display_end.strftime('%Y-%m-%d')}"
                )
                groups[label] = {
                    "df": group_df,
                    "number_of_strips": group_df[
                        strip_col
                    ].dropna().nunique(),
                    "label": label,
                    "period_start": week_start,
                    "period_end": week_end,
                }

            device_data[device] = {
                "df": device_df,
                "strip_col": strip_col,
                "groups": groups,
            }

        elif analysis_mode == "Monthly":
            device_df = device_df.copy()
            device_df["MonthPeriod"] = device_df["Data"].dt.to_period("M")

            groups = {}
            for month_period, group_df in device_df.groupby(
                "MonthPeriod",
                sort=True,
            ):
                label = month_period.strftime("%B %Y")
                groups[label] = {
                    "df": group_df,
                    "number_of_strips": group_df[
                        strip_col
                    ].dropna().nunique(),
                    "label": label,
                }

            device_data[device] = {
                "df": device_df,
                "strip_col": strip_col,
                "groups": groups,
            }

    return strip_col, devices_with_data


def extract_meslot_batch_code(value):
    """Extract the 3-digit batch code used in MESLotID."""
    match = re.search(r"(\d{3})(?=[A-Za-z])", str(value))
    return match.group(1) if match else None


def summarize_batch_group(device, batch_code, sequence, rows):
    """Summarize a contiguous group of MES lots."""
    rows_df = pd.DataFrame(rows)
    return {
        "USMDevice": device,
        "Batch": f"{batch_code}-{sequence}",
        "Start": rows_df["LotStart"].min(),
        "End": rows_df["LotEnd"].max(),
        "Lots": rows_df["MESLotID"].nunique(),
        "Strips": rows_df["StripCount"].sum(),
        "RecordCount": rows_df["RecordCount"].sum(),
        "MESLotID(s)": ", ".join(rows_df["MESLotID"].astype(str).tolist()),
    }


def build_batch_overview(source_df, start_value=None, end_value=None, devices=None):
    """Build suggested batch groups from the selected device/date population."""
    if source_df is None or source_df.empty:
        return pd.DataFrame()

    required = {"MESLotID", "USMDevice", "Data"}
    missing = required - set(source_df.columns)
    if missing:
        raise ValueError(
            "Batch Helper requires columns: " + ", ".join(sorted(missing))
        )

    work = source_df.dropna(
        subset=["Data", "MESLotID", "USMDevice"],
    ).copy()
    work["USMDevice"] = work["USMDevice"].astype(str).str.strip()

    if devices is not None:
        selected_devices = {str(d).strip() for d in devices if str(d).strip()}
        work = work[work["USMDevice"].isin(selected_devices)].copy()

    if start_value is not None and end_value is not None:
        start = pd.to_datetime(start_value).normalize()
        end = pd.to_datetime(end_value).normalize()
        if start > end:
            raise ValueError("Start date cannot be later than end date.")
        work = work[
            (work["Data"] >= start)
            & (work["Data"] < end + pd.Timedelta(days=1))
        ].copy()

    if work.empty:
        return pd.DataFrame()

    work["MESLotID"] = work["MESLotID"].astype(str).str.strip()
    work["BatchCode"] = work["MESLotID"].map(extract_meslot_batch_code)
    work = work[work["BatchCode"].notna()].copy()

    if work.empty:
        return pd.DataFrame()

    lot_rows = []
    for (device, batch_code, lot_id), group in work.groupby(
        ["USMDevice", "BatchCode", "MESLotID"],
        sort=False,
    ):
        lot_rows.append(
            {
                "USMDevice": device,
                "BatchCode": batch_code,
                "MESLotID": lot_id,
                "LotStart": group["Data"].min().normalize(),
                "LotEnd": group["Data"].max().normalize(),
                "StripCount": (
                    group["StripID"].dropna().astype(str).nunique()
                    if "StripID" in group.columns
                    else 0
                ),
                "RecordCount": len(group),
            }
        )

    lots = pd.DataFrame(lot_rows)
    if lots.empty:
        return lots

    batch_rows = []
    for (device, batch_code), group in lots.groupby(
        ["USMDevice", "BatchCode"],
        sort=False,
    ):
        group = group.sort_values(
            ["LotStart", "LotEnd", "MESLotID"],
            kind="stable",
        ).copy()

        batch_sequence = 0
        current_lots = []
        previous_end = None

        for _, row in group.iterrows():
            current_start = row["LotStart"]
            if (
                previous_end is None
                or current_start - previous_end >= pd.Timedelta(days=2)
            ):
                if current_lots:
                    batch_rows.append(
                        summarize_batch_group(
                            device,
                            batch_code,
                            batch_sequence,
                            current_lots,
                        )
                    )
                batch_sequence += 1
                current_lots = []

            current_lots.append(row.to_dict())
            previous_end = max(
                previous_end if previous_end is not None else row["LotEnd"],
                row["LotEnd"],
            )

        if current_lots:
            batch_rows.append(
                summarize_batch_group(
                    device,
                    batch_code,
                    batch_sequence,
                    current_lots,
                )
            )

    result = pd.DataFrame(batch_rows)
    if result.empty:
        return result

    result["BatchKey"] = (
        result["USMDevice"].astype(str)
        + " | "
        + result["Batch"].astype(str)
    )
    return result.sort_values(
        ["Start", "End"],
        ascending=[False, False],
        kind="stable",
    ).reset_index(drop=True)


def selected_batch_keys(batch_checkbox_map):
    """Return the currently checked BatchKey values."""
    return {
        key
        for key, checkbox in batch_checkbox_map.items()
        if checkbox.value
    }


def batch_lot_records(batch_overview_df, batch_keys):
    """Expand selected batch groups into individual device/lot records."""
    if batch_overview_df is None or batch_overview_df.empty:
        return []

    selected_rows = batch_overview_df[
        batch_overview_df["BatchKey"].astype(str).isin(
            {str(key) for key in batch_keys}
        )
    ]
    records = []
    for _, row in selected_rows.iterrows():
        for lot_id in str(row["MESLotID(s)"]).split(", "):
            records.append(
                {
                    "BatchKey": row["BatchKey"],
                    "USMDevice": str(row["USMDevice"]),
                    "Batch": str(row["Batch"]),
                    "MESLotID": str(lot_id),
                }
            )
    return records


def filter_by_selected_batches(dataframe, selected_batch_lots):
    """Filter an already device/date-filtered dataframe by selected lots."""
    if dataframe is None or dataframe.empty or not selected_batch_lots:
        return (
            dataframe.iloc[0:0].copy()
            if dataframe is not None
            else pd.DataFrame()
        )

    pairs = {
        (str(record["USMDevice"]).strip(), str(record["MESLotID"]).strip())
        for record in selected_batch_lots
    }
    device_values = dataframe["USMDevice"].astype(str).str.strip()
    lot_values = dataframe["MESLotID"].astype(str).str.strip()
    return dataframe[
        [(device, lot) in pairs for device, lot in zip(device_values, lot_values)]
    ].copy()
