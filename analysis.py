"""Core spatial and statistical analysis for Phase 10."""

from __future__ import annotations

import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import jensenshannon
from sklearn.cluster import DBSCAN

def plot_panel_map(grid, ax, panel_size, smooth=False, sigma=1.5):

    # --- build panel layout ---
    new_grid = []

    for b in range(4):
        start = b * panel_size
        end = start + panel_size

        new_grid.append(grid[:, start:end])

        # gap between panels
        if b < 3:
            new_grid.append(
                np.full((grid.shape[0], 2), 0)
            )

    new_grid = np.hstack(new_grid)

    max_A = np.max(new_grid)

    # --- optional smoothing ---
    if smooth:
        plot_grid = gaussian_filter(
            new_grid,
            sigma=sigma
        )

        im = ax.imshow(
            plot_grid,
            cmap="Reds",
            origin="lower",
            aspect="equal"
        )

    else:
        plot_grid = new_grid

        im = ax.imshow(
            plot_grid,
            cmap="Reds",
            origin="lower",
            vmax=max_A + 3,
            aspect="equal"
        )

    # --- draw panel grid lines ---
    current_x = 0

    for b in range(4):

        # vertical lines
        for x in range(panel_size + 1):
            ax.axvline(
                current_x + x - 0.5,
                color="black",
                linewidth=1
            )

        # horizontal lines
        for y in range(grid.shape[0] + 1):
            ax.axhline(
                y - 0.5,
                xmin=(current_x) / new_grid.shape[1],
                xmax=(current_x + panel_size) / new_grid.shape[1],
                color="black",
                linewidth=0.5
            )

        current_x += panel_size + 2

    # --- numbers inside grid ---
    for i in range(new_grid.shape[0]):
        for j in range(new_grid.shape[1]):

            if (
                not np.isnan(new_grid[i, j])
                and new_grid[i, j] > 0
            ):
                ax.text(
                    j,
                    i,
                    int(new_grid[i, j]),
                    ha="center",
                    va="center",
                    fontsize=8
                )

    # --- formatting ---
    ax.axis("off")
    ax.invert_xaxis()
    ax.invert_yaxis()

    return im


def plot_device_map(
    df,
    ax,
    device_name,
    device_config,
    row_mask=None,
    smooth=False,
    sigma=1.5
):

    # ---------------------------------------------------------
    # Get device configuration from app.py
    # ---------------------------------------------------------

    if device_name not in device_config:
        raise ValueError(
            f"Unknown device: {device_name}"
        )

    config = device_config[device_name]

    rows = config["rows"]
    cols = config["cols"]
    panel_size = config["panel_size"]

    # ---------------------------------------------------------
    # Create empty grid
    # ---------------------------------------------------------

    grid = np.zeros(
        (rows, cols)
    )

    # ---------------------------------------------------------
    # No filter -> use all rows
    # ---------------------------------------------------------

    if row_mask is None:

        x = (
            df["stripX"]
            .to_numpy()
            - 1
        )

        y = (
            df["stripY"]
            .to_numpy()
            - 1
        )

    else:

        x = (
            df.loc[
                row_mask,
                "stripX"
            ]
            .to_numpy()
            - 1
        )

        y = (
            df.loc[
                row_mask,
                "stripY"
            ]
            .to_numpy()
            - 1
        )

    # ---------------------------------------------------------
    # Accumulate counts
    # ---------------------------------------------------------

    np.add.at(
        grid,
        (y, x),
        1
    )

    # ---------------------------------------------------------
    # Use ORIGINAL plotting template
    # ---------------------------------------------------------

    return plot_panel_map(
        grid,
        ax,
        panel_size=panel_size,
        smooth=smooth,
        sigma=sigma
    )


def transform_panel_coordinates(df, device_name, device_config):
    df = df.copy()
    device_name = str(device_name)
    if device_name not in device_config:
        raise ValueError(f"No device configuration found for {device_name}.")

    cfg = device_config[device_name]
    rows = int(cfg["rows"])
    panel_size = int(cfg["panel_size"])

    if "stripX" not in df.columns or "stripY" not in df.columns:
        raise ValueError("Input data must contain stripX and stripY columns.")

    df["panel"] = ((df["stripX"] - 1) // panel_size).astype(int)
    df["panel_x"] = ((df["stripX"] - 1) - df["panel"] * panel_size)
    df["x_rel_panel"] = df["panel_x"] / (panel_size - 1) if panel_size > 1 else 0.0
    df["y_rel_panel"] = (df["stripY"] - 1) / (rows - 1) if rows > 1 else 0.0
    return df


def transform_panel_coordinates_multi(df, devices, device_config):
    """Normalize panel coordinates using each device's own geometry."""
    frames = []
    device_set = {str(d) for d in devices}
    work = df[df["USMDevice"].astype(str).isin(device_set)].copy()
    for device in devices:
        part = work[work["USMDevice"].astype(str) == str(device)].copy()
        if part.empty:
            continue
        frames.append(transform_panel_coordinates(part, device, device_config))
    if not frames:
        raise ValueError("No data available for the selected devices.")
    return pd.concat(frames, ignore_index=True)


def add_spatial_bins(df, grid_size=6):
    out = df.copy()
    out["row_bin"] = np.minimum(
        (out["y_rel_panel"] * grid_size).astype(int), grid_size - 1
    )
    out["col_bin"] = np.minimum(
        (out["x_rel_panel"] * grid_size).astype(int), grid_size - 1
    )
    out["bin_id"] = (
        out["panel"].astype(int).astype(str) + "_"
        + out["row_bin"].astype(str) + "_"
        + out["col_bin"].astype(str)
    )
    return out


def get_strip_sequence(df):
    return df.groupby("StripID")["Data"].min().sort_values().index.tolist()


def build_strip_bin_matrix(df):
    return (
        df.groupby(["StripID", "bin_id"])
        .size()
        .unstack(fill_value=0)
        .sort_index(axis=1)
    )


def build_distribution(matrix, strip_ids):
    counts = matrix.reindex(strip_ids, fill_value=0).sum(axis=0)
    total = counts.sum()
    return counts / total if total else counts.astype(float)


def calculate_jsd(p, q):
    if p.sum() == 0 or q.sum() == 0:
        return np.nan
    return jensenshannon(p, q, base=2) ** 2


def calculate_jsd_from_strips(matrix, strips_a, strips_b):
    return calculate_jsd(
        build_distribution(matrix, strips_a),
        build_distribution(matrix, strips_b),
    )


def calculate_jsd_contribution(p, q):
    m = (p + q) / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        return (
            np.where(p > 0, 0.5 * p * np.log2(p / m), 0)
            + np.where(q > 0, 0.5 * q * np.log2(q / m), 0)
        )


def plot_jsd_contribution_map(
    contribution_df,
    grid_size=6,
    top_n=10,
    title=None,
):
    """
    Shared JSD contribution map for Method 1 and Method 2.

    DISPLAY CONVENTION
    -------------------

    Panels are displayed in ONE horizontal row:

        Panel 4 | Panel 3 | Panel 2 | Panel 1

    Therefore Panel 1 is physically on the RIGHT.

    Within EACH panel:

        col_bin = 0  -> RIGHT
        col_bin ↑    -> LEFT

        row_bin = 0  -> TOP
        row_bin ↑    -> BOTTOM

    Internal panel values are NOT changed.

        internal 0 -> physical Panel 1
        internal 1 -> physical Panel 2
        internal 2 -> physical Panel 3
        internal 3 -> physical Panel 4

    This function only changes visualization.
    """

    from matplotlib import colors
    from matplotlib.patches import Rectangle

    # ========================================================
    # 1. VALIDATE INPUT
    # ========================================================

    if (
        contribution_df is None
        or contribution_df.empty
    ):
        print(
            "No JSD contribution data available."
        )
        return

    df = contribution_df.copy()

    # ========================================================
    # 2. IDENTIFY VALUE COLUMN
    # ========================================================

    if "signed_jsd_contribution" in df.columns:

        value_col = "signed_jsd_contribution"

    elif "mean_signed_jsd" in df.columns:

        value_col = "mean_signed_jsd"

    elif "mean_jsd" in df.columns:

        value_col = "mean_jsd"

    elif "jsd_contribution_pct" in df.columns:

        value_col = "jsd_contribution_pct"

    else:

        raise ValueError(
            "No supported JSD contribution "
            "column found."
        )

    # ========================================================
    # 3. RANK TOP N BINS
    # ========================================================
    #
    # Ranking is based on JSD contribution magnitude.
    #
    # Do NOT rank based on signed value.
    #
    # Example:
    #
    #     +5% -> rank higher
    #     -4% -> rank second
    #
    # ========================================================

    if "mean_jsd" in df.columns:

        rank_col = "mean_jsd"

    elif "jsd_contribution_pct" in df.columns:

        rank_col = "jsd_contribution_pct"

    else:

        rank_col = value_col

    df["_rank_magnitude"] = np.abs(
        pd.to_numeric(
            df[rank_col],
            errors="coerce",
        )
    )

    top = (
        df
        .sort_values(
            "_rank_magnitude",
            ascending=False,
        )
        .head(top_n)
        .copy()
        .reset_index(drop=True)
    )

    top["plot_rank"] = np.arange(
        1,
        len(top) + 1,
    )

    top_bins = set(
        top["bin_id"]
    )

    # ========================================================
    # 4. COLOR SCALE
    # ========================================================

    values = pd.to_numeric(
        df[value_col],
        errors="coerce",
    )

    finite_values = values[
        np.isfinite(values)
    ]

    if finite_values.empty:

        vmax = 1.0

    else:

        vmax = np.max(
            np.abs(
                finite_values
            )
        )

        if (
            not np.isfinite(vmax)
            or vmax <= 0
        ):
            vmax = 1.0

    norm = colors.TwoSlopeNorm(
        vmin=-vmax,
        vcenter=0,
        vmax=vmax,
    )

    # ========================================================
    # 6. PHYSICAL DISPLAY ORDER
    # ========================================================
    #
    # LEFT → RIGHT:
    #
    #     Panel 4 | Panel 3 | Panel 2 | Panel 1
    #
    # ========================================================

    physical_panel_order = [
        4,
        3,
        2,
        1,
    ]

    # ========================================================
    # 7. CREATE ONE HORIZONTAL ROW
    # ========================================================

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(16, 4.5),
        squeeze=False,
    )

    axes = axes[0]

    # ========================================================
    # 8. PLOT EACH PHYSICAL PANEL
    # ========================================================

    for plot_index, physical_panel in enumerate(
        physical_panel_order
    ):

        ax = axes[plot_index]

        # ----------------------------------------------------
        # Physical panel -> internal panel
        # ----------------------------------------------------

        internal_panel = (
            physical_panel - 1
        )

        panel_df = df[
            df["panel"].astype(int)
            ==
            internal_panel
        ].copy()

        # ====================================================
        # BUILD HEATMAP
        # ====================================================

        heat = np.full(
            (
                grid_size,
                grid_size,
            ),
            np.nan,
        )

        bin_matrix = np.empty(
            (
                grid_size,
                grid_size,
            ),
            dtype=object,
        )

        bin_matrix[:] = None

        for _, row in panel_df.iterrows():

            r = int(
                row["row_bin"]
            )

            c = int(
                row["col_bin"]
            )

            if not (
                0 <= r < grid_size
                and
                0 <= c < grid_size
            ):
                continue

            heat[
                r,
                c
            ] = row[
                value_col
            ]

            bin_matrix[
                r,
                c
            ] = row[
                "bin_id"
            ]

        # ====================================================
        # HEATMAP
        # ====================================================
        #
        # origin="upper"
        #
        #     row 0 -> TOP
        #     row 5 -> BOTTOM
        #
        # invert_xaxis()
        #
        #     col 0 -> RIGHT
        #     col 5 -> LEFT
        #
        # ====================================================

        ax.imshow(
            heat,
            origin="upper",
            cmap="RdBu_r",
            norm=norm,
            interpolation="nearest",
        )

        ax.invert_xaxis()

        # ====================================================
        # HIGHLIGHT TOP N
        # ====================================================

        for r in range(
            grid_size
        ):

            for c in range(
                grid_size
            ):

                if np.isnan(
                    heat[r, c]
                ):
                    continue

                bin_id = (
                    bin_matrix[
                        r,
                        c
                    ]
                )

                if (
                    bin_id
                    not in top_bins
                ):
                    continue

                selected = top[
                    top["bin_id"]
                    ==
                    bin_id
                ]

                if selected.empty:
                    continue

                selected_row = (
                    selected.iloc[0]
                )

                rank = int(
                    selected_row[
                        "plot_rank"
                    ]
                )

                # --------------------------------------------
                # Magnitude
                # --------------------------------------------

                if (
                    "mean_jsd"
                    in selected_row.index
                ):

                    magnitude = abs(
                        float(
                            selected_row[
                                "mean_jsd"
                            ]
                        )
                    )

                elif (
                    "jsd_contribution_pct"
                    in selected_row.index
                ):

                    magnitude = abs(
                        float(
                            selected_row[
                                "jsd_contribution_pct"
                            ]
                        )
                    )

                else:

                    magnitude = abs(
                        float(
                            selected_row[
                                value_col
                            ]
                        )
                    )

                # --------------------------------------------
                # Direction
                # --------------------------------------------

                if (
                    "mean_delta"
                    in selected_row.index
                ):

                    delta = (
                        selected_row[
                            "mean_delta"
                        ]
                    )

                elif (
                    "delta"
                    in selected_row.index
                ):

                    delta = (
                        selected_row[
                            "delta"
                        ]
                    )

                else:

                    delta = heat[
                        r,
                        c
                    ]

                if delta >= 0:

                    direction = "+"

                else:

                    direction = "-"

                # --------------------------------------------
                # Label
                # --------------------------------------------

                label = (
                    f"#{rank}\n"
                    f"{direction}"
                    f"{magnitude:.2f}%"
                )

                # =================================================
                # TEXT COLOR BASED ON BACKGROUND
                # =================================================

                rgba = plt.cm.RdBu_r(
                    norm(
                        heat[
                            r,
                            c
                        ]
                    )
                )

                brightness = (
                    0.299 * rgba[0]
                    +
                    0.587 * rgba[1]
                    +
                    0.114 * rgba[2]
                )

                if brightness < 0.55:

                    text_color = "white"

                else:

                    text_color = "black"

                ax.text(
                    c,
                    r,
                    label,
                    ha="center",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                    color=text_color,
                )

                # =================================================
                # RED BORDER
                # =================================================

                ax.add_patch(
                    Rectangle(
                        (
                            c - 0.5,
                            r - 0.5,
                        ),
                        1,
                        1,
                        fill=False,
                        edgecolor="red",
                        linewidth=2.5,
                    )
                )

        # ====================================================
        # 9. PANEL TITLE
        # ====================================================

        ax.set_title(
            f"Panel {physical_panel}",
            fontsize=11,
            fontweight="bold",
            pad=8,
        )

        ax.set_xticks([])
        ax.set_yticks([])

    # ========================================================
    # 13. FIGURE TITLE
    # ========================================================

    if title is None:

        title = (
            "JSD Contribution"
        )

    fig.suptitle(
        title,
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )

    # ========================================================
    # 14. LAYOUT
    # ========================================================

    plt.subplots_adjust(
        left=0.02,
        right=0.98,
        bottom=0.08,
        top=0.82,
        wspace=0.08,
    )

    # ========================================================
    # 14. LAYOUT
    # ========================================================

    plt.subplots_adjust(
        left=0.02,
        right=0.98,
        bottom=0.08,
        top=0.82,
        wspace=0.08,
    )

    # Return the exact notebook template as PNG bytes instead of
    # displaying it with plt.show(). The plotting template above is
    # intentionally unchanged.
    buffer = io.BytesIO()
    fig.savefig(
        buffer,
        format="png",
        dpi=160,
        bbox_inches="tight",
        facecolor="white",
        pad_inches=0.05,
    )
    plt.close(fig)
    return buffer.getvalue()


def build_jsd_contribution_table(matrix, strips_current, strips_reference):
    p = build_distribution(matrix, strips_current)
    q = build_distribution(matrix, strips_reference)
    contribution = calculate_jsd_contribution(p, q)
    total = contribution.sum()

    result = pd.DataFrame({
        "bin_id": matrix.columns,
        "current_prob": p,
        "reference_prob": q,
        "delta": p - q,
        "jsd_contribution": contribution,
    })
    result["jsd_contribution_pct"] = (
        result["jsd_contribution"] / total * 100 if total else 0
    )

    parts = result["bin_id"].str.split("_", expand=True)
    result["panel"] = parts[0].astype(int)
    result["row_bin"] = parts[1].astype(int)
    result["col_bin"] = parts[2].astype(int)

    return result.sort_values("jsd_contribution_pct", ascending=False).reset_index(drop=True)


def get_top_jsd_bins(contribution_df, top_n=10):
    if contribution_df is None or contribution_df.empty:
        return pd.DataFrame()
    out = contribution_df.copy()
    out["contribution_magnitude"] = out["jsd_contribution_pct"].abs()
    out = out.sort_values("contribution_magnitude", ascending=False).head(top_n).reset_index(drop=True)
    out["bin_rank"] = np.arange(len(out)) + 1
    return out


def permutation_jsd(
    df,
    strips_a,
    strips_b,
    n_perm=500,
    random_state=42,
):
    """Permutation null distribution for the observed Batch-B vs Batch-A JSD.

    The strip labels are shuffled while preserving the original sample sizes:
    each permutation randomly selects len(strips_b) strips as the current group,
    with the remaining strips forming the reference group.
    """
    strips_a = list(pd.Series(strips_a).dropna().unique())
    strips_b = list(pd.Series(strips_b).dropna().unique())
    all_strips = list(dict.fromkeys(strips_a + strips_b))

    if not strips_a or not strips_b:
        raise ValueError("Both reference and current strip groups are required for permutation JSD.")

    n_current = len(strips_b)
    if n_current >= len(all_strips):
        raise ValueError("Current group must contain fewer strips than the combined strip set.")

    matrix = build_strip_bin_matrix(df)
    rng = np.random.default_rng(random_state)
    values = np.full(int(n_perm), np.nan, dtype=float)

    for i in range(int(n_perm)):
        perm_current = rng.choice(all_strips, size=n_current, replace=False).tolist()
        perm_current_set = set(perm_current)
        perm_reference = [sid for sid in all_strips if sid not in perm_current_set]
        values[i] = calculate_jsd_from_strips(
            matrix,
            perm_reference,
            perm_current,
        )

    return values


def summarize_permutation_jsd(permutation_values, actual_jsd, p_threshold=5):
    """Summarize the permutation distribution using an upper-tail p percentage."""
    values = np.asarray(permutation_values, dtype=float)
    values = values[np.isfinite(values)]
    actual = float(actual_jsd)

    if values.size == 0 or not np.isfinite(actual):
        return {
            "actual_jsd": actual,
            "permutation_mean": np.nan,
            "permutation_std": np.nan,
            "p_pct": np.nan,
            "significant": False,
            "n_permutations": int(values.size),
        }

    # Add-one correction avoids a reported p-value of exactly 0%.
    exceed_count = int(np.sum(values >= actual))
    p_pct = (exceed_count + 1) / (values.size + 1) * 100.0

    return {
        "actual_jsd": actual,
        "permutation_mean": float(np.mean(values)),
        "permutation_std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        "p_pct": float(p_pct),
        "significant": bool(p_pct <= float(p_threshold)),
        "n_permutations": int(values.size),
    }


def _method2_contribution_map(result, top_n=10):
    contribution = result.get("contribution_df", pd.DataFrame()).copy()
    if contribution.empty:
        return None
    contribution["delta"] = contribution["current_prob"] - contribution["reference_prob"]
    contribution["sign"] = np.sign(contribution["delta"])
    contribution["signed_jsd_contribution"] = (
        contribution["jsd_contribution_pct"] * contribution["sign"]
    )
    stats = result.get("stats", {})
    actual = float(stats.get("actual_jsd", np.nan))
    title = "Batch B vs Batch A - JSD Contribution"
    if np.isfinite(actual):
        title = f"Batch B vs Batch A - JSD {actual:.2f}"
    return plot_jsd_contribution_map(
        contribution_df=contribution,
        grid_size=6,
        top_n=top_n,
        title=title,
    )


def analyze_spatial_clusters(
    df,
    eps=2.24,
    min_samples=3,
    min_cluster_size=3,
):
    """Detect spatial reject clusters using DBSCAN.

    Clustering is performed separately for each device, lot, strip,
    and panel using the raw stripX / stripY coordinates.
    """
    required = ["USMDevice", "MESLotID", "StripID", "stripX", "stripY"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for DBSCAN: {', '.join(missing)}")

    cluster_records = []
    strip_records = []
    cluster_points_records = []

    for (device, lot, strip_id), strip_data in df.groupby(
        ["USMDevice", "MESLotID", "StripID"], dropna=False
    ):
        total_rejects = len(strip_data)
        strip_cluster_records = []

        for panel, panel_data in strip_data.groupby("panel", dropna=False):
            panel_rejects = len(panel_data)
            if panel_rejects < min_samples:
                continue

            coordinates = panel_data[["stripX", "stripY"]].to_numpy(
                dtype=float,
                copy=False,
            )
            labels = DBSCAN(
                eps=eps,
                min_samples=min_samples,
            ).fit_predict(coordinates)

            cluster_ids, cluster_sizes = np.unique(
                labels[labels >= 0],
                return_counts=True,
            )

            for cluster_id, cluster_size in zip(cluster_ids, cluster_sizes):
                cluster_size = int(cluster_size)
                if cluster_size < min_cluster_size:
                    continue

                cluster_fraction = (
                    cluster_size / total_rejects
                    if total_rejects
                    else np.nan
                )
                panel_fraction = (
                    cluster_size / panel_rejects
                    if panel_rejects
                    else np.nan
                )

                cluster_record = {
                    "USMDevice": device,
                    "MESLotID": lot,
                    "StripID": strip_id,
                    "panel": int(panel),
                    "cluster_id": int(cluster_id),
                    "cluster_size": cluster_size,
                    "total_rejects": total_rejects,
                    "cluster_fraction": cluster_fraction,
                    "panel_rejects": panel_rejects,
                    "panel_fraction": panel_fraction,
                }
                cluster_records.append(cluster_record)
                strip_cluster_records.append(cluster_record)

                cluster_indices = np.flatnonzero(labels == cluster_id)
                cluster_coordinates = coordinates[cluster_indices]
                cluster_points_records.append(
                    pd.DataFrame({
                        "USMDevice": device,
                        "MESLotID": lot,
                        "StripID": strip_id,
                        "panel": int(panel),
                        "cluster_id": int(cluster_id),
                        "stripX": cluster_coordinates[:, 0],
                        "stripY": cluster_coordinates[:, 1],
                    })
                )

        if strip_cluster_records:
            largest = max(strip_cluster_records, key=lambda x: x["cluster_size"])
            strip_records.append({
                "USMDevice": device,
                "MESLotID": lot,
                "StripID": strip_id,
                "total_rejects": total_rejects,
                "number_of_clusters": len(strip_cluster_records),
                "largest_cluster_size": largest["cluster_size"],
                "largest_cluster_fraction": largest["cluster_fraction"],
                "largest_cluster_panel": largest["panel"],
                "largest_cluster_id": largest["cluster_id"],
            })

    cluster_df = pd.DataFrame(cluster_records)
    strip_summary = pd.DataFrame(strip_records)

    if cluster_points_records:
        cluster_points_df = pd.concat(cluster_points_records, ignore_index=True)
    else:
        cluster_points_df = pd.DataFrame(columns=[
            "USMDevice", "MESLotID", "StripID", "panel", "cluster_id", "stripX", "stripY"
        ])

    if not cluster_df.empty:
        cluster_df = (
            cluster_df.sort_values(["cluster_size", "cluster_fraction"], ascending=[False, False])
            .reset_index(drop=True)
        )
        cluster_df.insert(0, "rank", np.arange(1, len(cluster_df) + 1))

    if not strip_summary.empty:
        strip_summary = (
            strip_summary.sort_values(
                ["largest_cluster_size", "largest_cluster_fraction"],
                ascending=[False, False],
            )
            .reset_index(drop=True)
        )
        strip_summary.insert(0, "rank", np.arange(1, len(strip_summary) + 1))

    return cluster_df, strip_summary, cluster_points_df


def _ensure_cluster_panel_column(df, device_config):
    """Add the raw-coordinate panel index without changing stripX/stripY."""
    out = df.copy()

    if "USMDevice" not in out.columns:
        raise ValueError("USMDevice is required to determine panel geometry.")
    if "stripX" not in out.columns:
        raise ValueError("stripX is required to determine panel geometry.")

    if not isinstance(device_config, dict):
        raise ValueError("Device configuration is required for Method 1 DBSCAN.")

    panel_sizes = {}
    for device, cfg in device_config.items():
        try:
            panel_sizes[str(device)] = int(cfg["panel_size"])
        except (KeyError, TypeError, ValueError):
            continue

    device_series = out["USMDevice"].astype(str)
    panel_size_series = device_series.map(panel_sizes)
    strip_x = pd.to_numeric(out["stripX"], errors="coerce")

    out["panel"] = ((strip_x - 1) // panel_size_series)

    if out["panel"].isna().any():
        missing_devices = sorted(
            device_series[out["panel"].isna()].unique()
        )
        raise ValueError(
            f"No valid panel_size configuration found for: "
            f"{', '.join(missing_devices)}"
        )

    out["panel"] = out["panel"].astype(int)
    return out


def run_method1(df, devices=None, device_config=None, eps=2.24, min_samples=3, min_cluster_size=3):
    """Run Method 1 using DBSCAN spatial clustering."""
    if devices is None:
        devices = df["USMDevice"].dropna().astype(str).unique().tolist()
        data = df.copy()
    else:
        devices = [str(d) for d in devices]
        if not devices:
            raise ValueError("At least one device is required for Method 1.")
        data = df[df["USMDevice"].astype(str).isin(set(devices))].copy()
    if data.empty:
        raise ValueError("No data available for the selected devices.")

    data["stripX"] = pd.to_numeric(data["stripX"], errors="coerce")
    data["stripY"] = pd.to_numeric(data["stripY"], errors="coerce")
    data = data.dropna(subset=["USMDevice", "MESLotID", "StripID", "stripX", "stripY"]).copy()
    data = _ensure_cluster_panel_column(data, device_config)

    cluster_df, strip_summary, cluster_points_df = analyze_spatial_clusters(
        data, eps=eps, min_samples=min_samples, min_cluster_size=min_cluster_size
    )

    result = {
        "data": data,
        "cluster_df": cluster_df,
        "strip_summary": strip_summary,
        "cluster_point_df": cluster_points_df,
        "devices": devices,
        "device_config": device_config,
        "eps": eps,
        "min_samples": min_samples,
        "min_cluster_size": min_cluster_size,
    }

    if not strip_summary.empty:
        top = strip_summary.iloc[0]
        result["plot_strip_id"] = top["StripID"]
        result["plot_device"] = str(top["USMDevice"])

    return result


def plot_dbscan_clusters(df, cluster_points_df, strip_id, device_name, ax, device_config=None):
    """Plot a strip reject map and overlay its meaningful DBSCAN clusters."""
    default_configs = {
        "SOLARIS-XRF3": (17, 76, 19),
        "VEGA": (15, 64, 16),
        "SQUID-ICD82PRV": (17, 72, 18),
    }
    if isinstance(device_config, dict) and device_name in device_config:
        cfg = device_config[device_name]
        rows = int(cfg["rows"])
        cols = int(cfg["cols"])
        panel_size = int(cfg["panel_size"])
    elif device_name in default_configs:
        rows, cols, panel_size = default_configs[device_name]
    else:
        raise ValueError(f"No device geometry found for {device_name}.")

    strip_df = df[
        df["StripID"].eq(strip_id) & df["USMDevice"].astype(str).eq(str(device_name))
    ].copy()
    grid = np.zeros((rows, cols))
    x = pd.to_numeric(strip_df["stripX"], errors="coerce").to_numpy() - 1
    y = pd.to_numeric(strip_df["stripY"], errors="coerce").to_numpy() - 1
    valid = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (x < cols) & (y >= 0) & (y < rows)
    np.add.at(grid, (y[valid].astype(int), x[valid].astype(int)), 1)

    new_grid = []
    for panel in range(4):
        start = panel * panel_size
        end = min(start + panel_size, cols)
        new_grid.append(grid[:, start:end])
        if panel < 3:
            new_grid.append(np.zeros((rows, 2)))
    new_grid = np.hstack(new_grid)

    max_count = np.max(new_grid) if new_grid.size else 0
    ax.imshow(new_grid, cmap="Reds", origin="lower", vmax=max_count + 3, aspect="equal")

    current_x = 0
    for panel in range(4):
        width = min(panel_size, max(cols - panel * panel_size, 0))
        if width <= 0:
            break
        for xx in range(width + 1):
            ax.axvline(current_x + xx - 0.5, color="black", linewidth=1)
        for yy in range(rows + 1):
            ax.axhline(yy - 0.5, xmin=current_x / new_grid.shape[1],
                        xmax=(current_x + width) / new_grid.shape[1],
                        color="black", linewidth=0.5)
        current_x += width + 2

    cluster_data = cluster_points_df[
        cluster_points_df["StripID"].eq(strip_id) &
        cluster_points_df["USMDevice"].astype(str).eq(str(device_name))
    ].copy()

    def raw_x_to_plot_x(raw_x):
        x0 = int(raw_x) - 1
        panel = x0 // panel_size
        local_x = x0 % panel_size
        return panel * (panel_size + 2) + local_x

    if not cluster_data.empty:
        # Cluster IDs restart from 0 for every panel, so cluster_id alone
        # cannot be used to assign colours.  Use (panel, cluster_id) as the
        # unique cluster key so clusters in different panels get different
        # colours within the same strip.
        cluster_keys = (
            cluster_data[["panel", "cluster_id"]]
            .drop_duplicates()
            .sort_values(["panel", "cluster_id"])
            .itertuples(index=False, name=None)
        )
        cluster_keys = list(cluster_keys)

        # Use the 20-colour categorical palette.  If a strip contains more
        # than 20 clusters, colours cycle only after all 20 colours are used.
        palette = plt.cm.tab20(np.arange(20))
        colour_map = {
            key: palette[i % 20]
            for i, key in enumerate(cluster_keys)
        }

        for _, point in cluster_data.iterrows():
            key = (point["panel"], point["cluster_id"])
            colour = colour_map[key]
            plot_x = raw_x_to_plot_x(point["stripX"])
            plot_y = int(point["stripY"]) - 1
            ax.add_patch(plt.Rectangle(
                (plot_x - 0.5, plot_y - 0.5), 1, 1,
                facecolor=colour, edgecolor="black", linewidth=1, zorder=8
            ))

    ax.set_title(f"Method 1 — DBSCAN Clusters | {device_name} | Strip: {strip_id}",
                 fontsize=12, fontweight="bold")
    ax.axis("off")
    ax.invert_xaxis()
    ax.invert_yaxis()
    return ax


def create_method1_plot(result, device=None, criteria=None, strip_id=None, plot_df=None):
    """Return a DBSCAN strip map as PNG bytes.

    If ``strip_id`` is supplied, that user-selected strip is plotted.
    Otherwise the highest-ranked strip in ``strip_summary`` is used.
    ``plot_df`` can be supplied separately so the background reject map can
    use the full date-range data even when an analysis filter was applied.
    """
    cluster_points_df = result.get("cluster_point_df", pd.DataFrame())
    strip_summary = result.get("strip_summary", pd.DataFrame())
    data = plot_df if plot_df is not None else result.get("data", pd.DataFrame())

    if strip_id is None:
        if strip_summary.empty:
            return None
        top = strip_summary.iloc[0]
        strip_id = top["StripID"]
        device_name = str(top["USMDevice"])
    else:
        if device is None:
            if strip_summary.empty:
                return None
            matches = strip_summary[strip_summary["StripID"].astype(str).eq(str(strip_id))]
            if matches.empty:
                return None
            device_name = str(matches.iloc[0]["USMDevice"])
        else:
            device_name = str(device)

    fig, ax = plt.subplots(figsize=(12, 6.5))
    plot_dbscan_clusters(
        data, cluster_points_df, strip_id, device_name, ax,
        device_config=result.get("device_config"),
    )
    # Total reject count for the selected strip (using the same data
    # passed to the plot, i.e. the current date-range/filter selection).
    if data is not None and not data.empty and "StripID" in data.columns:
        strip_mask = (
            data["StripID"].astype(str).eq(str(strip_id))
            & data["USMDevice"].astype(str).eq(str(device_name))
        )
        total_rejects = int(strip_mask.sum())
    else:
        total_rejects = 0

    title = (
        f"Method 1 — DBSCAN | {device_name} | Strip: {strip_id} "
        f"| Total Rejects: {total_rejects:,}"
    )
    if criteria is not None:
        title += f" | Criteria {criteria}"
    ax.set_title(title, fontsize=15, fontweight="bold")
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return buffer.getvalue()


def run_method2(df, devices, device_config, strips_a, strips_b, n_perm=500, random_state=42, grid_size=6, top_n=10, p_threshold=5):
    devices = [str(d) for d in devices]
    data = transform_panel_coordinates_multi(df, devices, device_config)
    data = add_spatial_bins(data, grid_size)
    matrix = build_strip_bin_matrix(data)
    actual_jsd = calculate_jsd_from_strips(matrix, strips_b, strips_a)
    permutation_values = permutation_jsd(data, strips_a, strips_b, n_perm=n_perm, random_state=random_state)
    stats = summarize_permutation_jsd(permutation_values, actual_jsd, p_threshold=p_threshold)
    contribution_df = build_jsd_contribution_table(matrix, strips_b, strips_a)
    contribution_df["delta"] = contribution_df["current_prob"] - contribution_df["reference_prob"]
    contribution_df["sign"] = np.sign(contribution_df["delta"])
    contribution_df["signed_jsd_contribution"] = contribution_df["jsd_contribution_pct"] * contribution_df["sign"]
    top_bins = get_top_jsd_bins(contribution_df, top_n=top_n)
    actual_rejects = trace_method2_rejects(data, strips_b, top_bins)
    result = {
        "data": data,
        "matrix": matrix,
        "reference_strips": list(strips_a),
        "current_strips": list(strips_b),
        "actual_jsd": actual_jsd,
        "permutation_jsd": permutation_values,
        "stats": stats,
        "contribution_df": contribution_df,
        "top_bins": top_bins,
        "actual_rejects": actual_rejects,
        "devices": devices,
    }
    result["contribution_map_png"] = _method2_contribution_map(result, top_n=top_n)
    return result


def trace_method2_rejects(data, current_strips, top_bins):
    if top_bins is None or top_bins.empty:
        return pd.DataFrame()
    current_df = data[
        data["StripID"].isin(current_strips)
        & data["UserRejectCode"].astype(str).str.strip().eq("TA")
    ].copy()
    out = current_df.merge(top_bins[["bin_id", "bin_rank", "jsd_contribution_pct"]], on="bin_id", how="inner")
    if out.empty:
        return pd.DataFrame()
    out["method"] = "Method 2"
    out["reject_role"] = "Current batch"
    preferred = [
        "method", "reject_role", "bin_rank", "bin_id", "jsd_contribution_pct",
        "MESLotID", "StripID", "USMDevice", "stripX", "stripY", "Data", "UserRejectCode", "MachineRejectCode"
    ]
    cols = [c for c in preferred if c in out.columns] + [c for c in out.columns if c not in preferred]
    return out[cols].sort_values(["bin_rank", "StripID"], kind="stable").reset_index(drop=True)


def create_method2_plot(result, device, criteria=None):
    """Return Method 2 permutation-JSD distribution as PNG bytes."""
    values = np.asarray(result.get("permutation_jsd", []), dtype=float)
    values = values[np.isfinite(values)]
    stats = result.get("stats", {})
    actual = float(stats.get("actual_jsd", np.nan))
    mean = float(stats.get("permutation_mean", np.nan))
    fig, ax = plt.subplots(figsize=(12, 6.5))
    if len(values):
        ax.hist(values, bins=30, alpha=0.75, label="Permutation JSD")
    if np.isfinite(mean):
        ax.axvline(mean, linestyle="--", linewidth=1.2, label=f"Permutation mean = {mean:.6f}")
    if np.isfinite(actual):
        ax.axvline(actual, linestyle="-", linewidth=1.8, label=f"Actual JSD = {actual:.6f}")
    p_pct = stats.get("p_pct", np.nan)
    significance = "SIGNIFICANT" if stats.get("significant", False) else "NOT SIGNIFICANT"
    title = f"Method 2 — Permutation JSD | {device}"
    if criteria is not None:
        title += f" | Criteria {criteria}"
    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.set_xlabel("JSD")
    ax.set_ylabel("Frequency")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    if np.isfinite(p_pct):
        ax.text(0.98, 0.97, f"p% = {p_pct:.2f}\n{significance}", transform=ax.transAxes,
                ha="right", va="top", bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="black"))
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return buffer.getvalue()

