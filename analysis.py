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
from matplotlib.collections import LineCollection

def _build_panel_layout(grid, panel_size):
    """Build the plotting grid without changing the original panel layout."""
    rows, cols = grid.shape
    parts = [
        grid[:, start:start + panel_size]
        for start in range(0, cols, panel_size)
    ]

    if len(parts) != 4 or any(part.shape[1] != panel_size for part in parts):
        raise ValueError(
            f"Expected four panels of width {panel_size}, got grid shape {grid.shape}."
        )

    gap = np.zeros((rows, 2), dtype=grid.dtype)
    result = [parts[0]]
    for panel in parts[1:]:
        result.extend((gap, panel))

    return np.hstack(result)


def _draw_panel_grid(ax, rows, panel_size, total_width):
    segments = []
    current_x = 0

    for _ in range(4):
        for x in range(panel_size + 1):
            x_pos = current_x + x - 0.5
            segments.append(
                [(x_pos, -0.5), (x_pos, rows - 0.5)]
            )

        for y in range(rows + 1):
            y_pos = y - 0.5
            segments.append(
                [
                    (current_x - 0.5, y_pos),
                    (current_x + panel_size - 0.5, y_pos),
                ]
            )

        current_x += panel_size + 2

    ax.add_collection(
        LineCollection(
            segments,
            colors="black",
            linewidths=(
                [1.0] * (panel_size + 1) + [0.5] * (rows + 1)
            ) * 4,
            antialiased=False,
        )
    )


def _draw_positive_labels(ax, grid):
    """Draw the same positive-cell labels while skipping empty cells."""
    occupied = np.argwhere(
        np.isfinite(grid) & (grid > 0)
    )

    for row, col in occupied:
        ax.text(
            int(col),
            int(row),
            int(grid[row, col]),
            ha="center",
            va="center",
            fontsize=8,
        )


def plot_panel_map(grid, ax, panel_size, smooth=False, sigma=1.5):
    new_grid = _build_panel_layout(
        np.asarray(grid),
        int(panel_size),
    )

    max_A = np.max(new_grid)

    if smooth:
        plot_grid = gaussian_filter(
            new_grid,
            sigma=sigma,
        )

        im = ax.imshow(
            plot_grid,
            cmap="Reds",
            origin="lower",
            aspect="equal",
        )
    else:
        plot_grid = new_grid

        im = ax.imshow(
            plot_grid,
            cmap="Reds",
            origin="lower",
            vmax=max_A + 3,
            aspect="equal",
        )

    _draw_panel_grid(
        ax,
        rows=new_grid.shape[0],
        panel_size=int(panel_size),
        total_width=new_grid.shape[1],
    )

    _draw_positive_labels(
        ax,
        new_grid,
    )

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
    if device_name not in device_config:
        raise ValueError(
            f"Unknown device: {device_name}"
        )

    config = device_config[device_name]

    rows = int(config["rows"])
    cols = int(config["cols"])
    panel_size = int(config["panel_size"])

    if row_mask is None:
        work = df
    else:
        work = df.loc[row_mask]

    x = work["stripX"].to_numpy(dtype=np.intp, copy=False) - 1
    y = work["stripY"].to_numpy(dtype=np.intp, copy=False) - 1

    flat_indices = y * cols + x
    grid = np.bincount(
        flat_indices,
        minlength=rows * cols,
    ).reshape(rows, cols)

    return plot_panel_map(
        grid,
        ax,
        panel_size=panel_size,
        smooth=smooth,
        sigma=sigma,
    )

def transform_panel_coordinates(df, device_name, device_config):
    df = df.copy()
    device_name = str(device_name)
    if device_name not in device_config:
        raise ValueError(f"No device configuration found for {device_name}.")

    cfg = device_config[device_name]
    rows = int(cfg["rows"])
    panel_size = int(cfg["panel_size"])

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
    from matplotlib import colors
    from matplotlib.patches import Rectangle



    if (
        contribution_df is None
        or contribution_df.empty
    ):
        print(
            "No JSD contribution data available."
        )
        return

    df = contribution_df.copy()

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

    physical_panel_order = [
        4,
        3,
        2,
        1,
    ]

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(16, 4.5),
        squeeze=False,
    )

    axes = axes[0]


    for plot_index, physical_panel in enumerate(
        physical_panel_order
    ):

        ax = axes[plot_index]

        internal_panel = (
            physical_panel - 1
        )

        panel_df = df[
            df["panel"].astype(int)
            ==
            internal_panel
        ].copy()

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

                label = (
                    f"#{rank}\n"
                    f"{direction}"
                    f"{magnitude:.2f}%"
                )


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

        ax.set_title(
            f"Panel {physical_panel}",
            fontsize=11,
            fontweight="bold",
            pad=8,
        )

        ax.set_xticks([])
        ax.set_yticks([])

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

    plt.subplots_adjust(
        left=0.02,
        right=0.98,
        bottom=0.08,
        top=0.82,
        wspace=0.08,
    )

    plt.subplots_adjust(
        left=0.02,
        right=0.98,
        bottom=0.08,
        top=0.82,
        wspace=0.08,
    )

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

    if not isinstance(device_config, dict):
        raise ValueError("Device configuration is required for Method 1 DBSCAN.")

    panel_sizes = {}
    for device, cfg in device_config.items():
        try:
            panel_sizes[str(device)] = int(cfg["panel_size"])
        except (KeyError, TypeError, ValueError):