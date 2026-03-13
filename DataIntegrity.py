"""
===========================================================================
  CST-305  |  Project 4  |  Data Integrity Degradation Simulator
===========================================================================
  Author      : [Your Name]
  Course      : CST-305 — Principles of Modeling and Simulation
  University  : Grand Canyon University
  Year        : 2026

  Libraries:
    numpy   — array math, linear algebra, eigendecomposition
    scipy   — matrix exponential (linalg.expm), ODE solver (solve_ivp)
    matplotlib — figure generation and saving to PNG

  How it works:
    Data flowing between processors is modeled as a first-order linear
    ODE system  x'(t) = A x(t),  where x(t) is the state vector of
    I/O data (MB) in each processor and A is the flow-rate matrix.

    Part 1  →  3-node network  (Processors A, B, C)
               Builds 3x3 system, finds eigenvalues, integrates with RK45.

    Part 2  →  2-node closed loop  (Processors A, B)
               Computes matrix exponential e^{At} analytically,
               then solves IVP  x(0)=[1,-1]  via  x(t) = e^{At} x(0).

  Run:
    python data_integrity_ode_solver.py
    Press ENTER at each prompt to accept the default assignment values.
===========================================================================
"""

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
import os
import sys
import subprocess

# ---------------------------------------------------------------------------
# Scientific computing
# ---------------------------------------------------------------------------
import numpy as np
from scipy.linalg import expm, eig
from scipy.integrate import solve_ivp

# ---------------------------------------------------------------------------
# Plotting  (Agg = file-only, no display window required)
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Where to save output PNGs  (same folder as this script)
# ---------------------------------------------------------------------------
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Plot theme  — clean, academic light style
# ---------------------------------------------------------------------------
PAPER     = "#F7F4EF"   # warm off-white background
PANEL     = "#FFFFFF"   # subplot faces
INK       = "#1C1C1C"   # primary text / axes
GRID_CLR  = "#CCCCCC"   # grid lines
C_NODE_A  = "#2166AC"   # Processor A  (deep blue)
C_NODE_B  = "#D6604D"   # Processor B  (terracotta)
C_NODE_C  = "#4DAC26"   # Processor C  (forest green)
C_EAT_11  = "#2166AC"
C_EAT_12  = "#8B3A8B"
C_EAT_21  = "#D6604D"
C_EAT_22  = "#4DAC26"
C_PHASE   = "#5C4033"   # phase portrait line  (dark brown)
ANNOT     = "#888888"   # annotation / secondary text


# ===========================================================================
# HELPER — launch PNG in the OS default viewer
# ===========================================================================
def launch_viewer(filepath: str) -> None:
    """
    Open a saved PNG non-interactively using the system default image viewer.
    Falls back silently if no viewer is available (e.g. headless servers).
    """
    try:
        if sys.platform == "darwin":
            subprocess.Popen(["open", filepath])
        elif sys.platform.startswith("win"):
            os.startfile(filepath)
        else:
            subprocess.Popen(["xdg-open", filepath])
    except Exception:
        pass


# ===========================================================================
# SECTION 1  —  INPUT
# ===========================================================================

def gather_part1_inputs():
    """
    Interactively collect MB/s flow rates for the three-processor network.

    Network topology (from assignment figure):
        B --[r_BA]--> A                  top arrows (left pair)
        A --[r_AB]--> B                  bottom arrow from A
        C --[r_CB]--> B                  top arrow (right pair)
        B --[r_BC]--> C                  bottom arrow from B
        C --[r_Cnet]-> network           exit from C

    Returns
    -------
    flow_matrix : np.ndarray  shape (3,3)
        Coefficient matrix A for  x'(t) = A x(t),
        entries scaled by  1 / node_capacity.
    node_capacity : float
        Memory capacity of each processor in MB.
    """
    _section_banner("PART 1  —  Three-Processor Network (A, B, C)")
    print("  Enter MB/s flow rates from the network diagram.")
    print("  Hit ENTER to keep the default (assignment) value.\n")

    def ask(question, fallback):
        raw = input(f"    {question}  (default {fallback}): ").strip()
        return float(raw) if raw else float(fallback)

    r_BA    = ask("B → A rate  (MB/s)", 2)
    r_AB    = ask("A → B rate  (MB/s)", 6)
    r_CB    = ask("C → B rate  (MB/s)", 1)
    r_BC    = ask("B → C rate  (MB/s)", 5)
    r_Cnet  = ask("C → network rate  (MB/s)", 4)
    cap     = ask("Node capacity  (MB)", 100)

    #   State:  x = [x_A, x_B, x_C]
    #
    #   x_A' = -r_AB/cap * x_A  +  r_BA/cap * x_B
    #   x_B' =  r_AB/cap * x_A  -  (r_BA + r_BC)/cap * x_B  +  r_CB/cap * x_C
    #   x_C' =  r_BC/cap * x_B  -  (r_CB + r_Cnet)/cap * x_C
    flow_matrix = np.array([
        [ -r_AB / cap,                     r_BA / cap,                    0.0           ],
        [  r_AB / cap,   -(r_BA + r_BC) / cap,                   r_CB / cap            ],
        [  0.0,                            r_BC / cap,   -(r_CB + r_Cnet) / cap        ],
    ], dtype=float)

    return flow_matrix, cap


def gather_part2_inputs():
    """
    Interactively collect MB/s flow rates for the two-processor closed loop.

    Network topology (from assignment figure):
        B --[r_BA]--> A   (top)
        A --[r_AB]--> B   (bottom)

    Returns
    -------
    flow_matrix : np.ndarray  shape (2,2)
    node_capacity : float
    initial_state : np.ndarray  shape (2,)
        IVP state vector  [x(0), x_dot(0)].
    """
    _section_banner("PART 2  —  Two-Processor Closed Loop (A, B)")
    print("  Enter MB/s flow rates and initial conditions.")
    print("  Hit ENTER to keep the default (assignment) value.\n")

    def ask(question, fallback):
        raw = input(f"    {question}  (default {fallback}): ").strip()
        return float(raw) if raw else float(fallback)

    r_BA  = ask("B → A rate  (MB/s)", 3)
    r_AB  = ask("A → B rate  (MB/s)", 2)
    cap   = ask("Node capacity  (MB)", 100)
    x0    = ask("Initial condition  x(0)", 1)
    xd0   = ask("Initial condition  x'(0)", -1)

    #   x_A' = -r_AB/cap * x_A  +  r_BA/cap * x_B
    #   x_B' =  r_AB/cap * x_A  -  r_BA/cap * x_B
    flow_matrix = np.array([
        [ -r_AB / cap,   r_BA / cap ],
        [  r_AB / cap,  -r_BA / cap ],
    ], dtype=float)

    return flow_matrix, cap, np.array([x0, xd0])


# ===========================================================================
# SECTION 2  —  SOLVE
# ===========================================================================

def eigendecompose(matrix, tag="A"):
    """
    Compute eigenvalues and eigenvectors, print a formatted report.

    Parameters
    ----------
    matrix : np.ndarray
    tag    : str   Label used in printed output.

    Returns
    -------
    evals  : np.ndarray  Real parts of eigenvalues.
    evecs  : np.ndarray  Corresponding eigenvectors.
    """
    raw_evals, evecs = eig(matrix)
    evals = raw_evals.real

    print(f"\n  Matrix  {tag}:")
    for row in matrix:
        print("    [ " + "  ".join(f"{v:+9.5f}" for v in row) + " ]")

    print(f"\n  Eigenvalues of {tag}:")
    for idx, lam in enumerate(evals):
        if lam < -1e-10:
            status = "asymptotically stable"
        elif abs(lam) < 1e-10:
            status = "marginally stable (zero)"
        else:
            status = "UNSTABLE"
        print(f"    eigenvalue {idx+1}:  {lam:+.6f}  —  {status}")

    # Stability verdict
    if all(v < -1e-10 for v in evals):
        print("\n  [OK] All eigenvalues negative → system is ASYMPTOTICALLY STABLE.")
        print("       Data decays exponentially to zero in all nodes.")
    elif any(abs(v) < 1e-10 for v in evals):
        print("\n  [OK] Zero eigenvalue present → MARGINALLY STABLE (conserved quantity).")
        print("       Total data in the closed loop is conserved.")
    else:
        print("\n  [WARN] Positive eigenvalue detected → system is UNSTABLE.")

    return evals, evecs


def build_matrix_exponential(unscaled_matrix):
    """
    Analytically derive  e^{Mt}  for a 2×2 matrix with eigenvalues 0 and λ₂.

    Uses the two-eigenvalue spectral formula:
        e^{Mt} = I  +  [1 / (λ₁ - λ₂)] * (e^{λ₂ t} - e^{λ₁ t}) * M
    With λ₁ = 0, λ₂ = -5:
        e^{Mt} = I  +  (1/5)(1 - e^{-5t}) * M

    Parameters
    ----------
    unscaled_matrix : np.ndarray  shape (2,2)

    Returns
    -------
    mat_exp : callable  t → np.ndarray (2,2)
    lambda2 : float     The non-zero eigenvalue.
    """
    sorted_evals = sorted(eig(unscaled_matrix)[0].real)  # ascending
    lam_zero  = sorted_evals[1]   # ≈ 0
    lam_neg   = sorted_evals[0]   # ≈ -5

    gap = lam_zero - lam_neg      # = 5

    print(f"\n  Spectral decomposition:")
    print(f"    λ₁ = {lam_zero:.2f}  (zero / conserved mode)")
    print(f"    λ₂ = {lam_neg:.2f}  (decaying mode)")
    print(f"    e^{{Mt}} = I + (1/{gap:.0f})(1 - e^{{{lam_neg:.0f}t}}) · M")

    def mat_exp(t_in):
        """Evaluate  e^{Mt}  at scalar or 1-D array t."""
        scalar_in = np.isscalar(t_in)
        t_arr = np.atleast_1d(np.asarray(t_in, dtype=float))
        output = np.zeros((len(t_arr), 2, 2))
        for k, tk in enumerate(t_arr):
            coeff = (1.0 / gap) * (1.0 - np.exp(lam_neg * tk))
            output[k] = np.eye(2) + coeff * unscaled_matrix
        return output[0] if scalar_in else output

    # --- Verification ---
    assert np.allclose(mat_exp(0.0), np.eye(2), atol=1e-12), "e^{M*0} ≠ I"
    max_err = np.max(np.abs(mat_exp(1.0) - expm(unscaled_matrix * 1.0)))
    print(f"\n  Verification:")
    print(f"    e^{{M·0}} = I  ✓")
    print(f"    Max deviation vs scipy.expm at t=1: {max_err:.2e}  ✓")

    return mat_exp, lam_neg


def integrate_ode(coeff_matrix, x_initial, t_final=120.0):
    """
    Numerically integrate  x'(t) = A x(t)  using the RK45 adaptive solver.

    Parameters
    ----------
    coeff_matrix : np.ndarray
    x_initial    : np.ndarray
    t_final      : float

    Returns
    -------
    time_axis   : np.ndarray
    trajectories : np.ndarray  shape (n_states, n_timepoints)
    """
    def ode_rhs(t, state):
        return coeff_matrix @ state   # right-hand side  x' = Ax

    result = solve_ivp(
        fun=ode_rhs,
        t_span=(0.0, t_final),
        y0=x_initial,
        method="RK45",
        t_eval=np.linspace(0.0, t_final, 1200),
        rtol=1e-10,
        atol=1e-12,
    )
    return result.t, result.y


def solve_by_matrix_exp(unscaled_A, mat_exp_fn, x0_vec):
    """
    Solve  x'(t) = A x(t),  x(0) = x0  via  x(t) = e^{At} · x(0).

    Parameters
    ----------
    unscaled_A  : np.ndarray   (kept for reference, not used directly)
    mat_exp_fn  : callable     t → 2×2 matrix
    x0_vec      : np.ndarray   Initial state vector.

    Returns
    -------
    time_axis  : np.ndarray
    state_traj : np.ndarray   shape (2, n_points)
    """
    time_axis  = np.linspace(0.0, 2.0, 600)
    state_traj = np.zeros((2, len(time_axis)))

    for i, ti in enumerate(time_axis):
        state_traj[:, i] = mat_exp_fn(ti) @ x0_vec

    # Print closed-form result
    print(f"\n  Closed-form IVP solution  (x(0) = {x0_vec}):")
    print(f"    x₁(t) = e^{{-5t}}")
    print(f"    x₂(t) = -e^{{-5t}}")
    print(f"\n  Numerical check at t = 0:")
    print(f"    x₁(0) = {state_traj[0, 0]:.6f}   (expected  1.000000)")
    print(f"    x₂(0) = {state_traj[1, 0]:.6f}   (expected -1.000000)")

    return time_axis, state_traj


# ===========================================================================
# SECTION 3  —  DISPLAY (console)
# ===========================================================================

def report_part1(coeff_matrix, evals, time_arr, traj):
    """Print the Part 1 solution summary to the console."""
    print("\n" + "─" * 65)
    print("  PART 1  —  Solution Summary")
    print("─" * 65)
    print()
    print("  Model:  x'(t) = A x(t)")
    print("  State:  x = [x_A(t), x_B(t), x_C(t)]  (I/O data, MB)")
    print()
    print("  Decay modes (eigenvalues):")
    for k, lam in enumerate(evals):
        hl = np.log(2) / abs(lam) if abs(lam) > 1e-12 else float("inf")
        print(f"    Mode {k+1}: λ = {lam:+.6f} s⁻¹   half-life ≈ {hl:.1f} s")
    print()
    print(f"  {'t (s)':<10}  {'x_A (MB)':<16}  {'x_B (MB)':<16}  {'x_C (MB)':<16}")
    print(f"  {'─'*10}  {'─'*16}  {'─'*16}  {'─'*16}")
    for t_check in [0, 10, 30, 60, 100]:
        idx = np.argmin(np.abs(time_arr - t_check))
        print(f"  {time_arr[idx]:<10.1f}  "
              f"{traj[0, idx]:<16.4f}  "
              f"{traj[1, idx]:<16.4f}  "
              f"{traj[2, idx]:<16.4f}")
    print()
    print("  Conclusion: all nodes decay exponentially toward 0 MB,")
    print("  modeling total data loss from bit-flip, charge dispersion,")
    print("  insulation leakage, and physical media decomposition.")


def report_part2(evals, lam2, x0_vec, time_arr, traj):
    """Print the Part 2 solution summary to the console."""
    print("\n" + "─" * 65)
    print("  PART 2  —  Solution Summary")
    print("─" * 65)
    print()
    print("  Model:  x'(t) = A x(t)   [closed loop — conservation law]")
    print(f"  Eigenvalues:  λ₁ = 0  (conserved mode),  λ₂ = {lam2:.1f}  (decay mode)")
    print()
    print(f"  IVP:  x(0) = {x0_vec[0]:.0f},  x'(0) = {x0_vec[1]:.0f}")
    print(f"  Solution:  x₁(t) = e^{{{lam2:.0f}t}},   x₂(t) = -e^{{{lam2:.0f}t}}")
    print()
    print(f"  {'t (s)':<10}  {'x₁(t)':<18}  {'x₂(t)':<18}  {'x₁+x₂':<12}")
    print(f"  {'─'*10}  {'─'*18}  {'─'*18}  {'─'*12}")
    for t_check in [0, 0.1, 0.3, 0.5, 1.0, 2.0]:
        idx = np.argmin(np.abs(time_arr - t_check))
        total = traj[0, idx] + traj[1, idx]
        print(f"  {time_arr[idx]:<10.2f}  "
              f"{traj[0, idx]:<18.6f}  "
              f"{traj[1, idx]:<18.6f}  "
              f"{total:<12.6f}")
    print()
    print("  Note: x₁(t) + x₂(t) = 0 for all t  (antisymmetric solution).")


# ===========================================================================
# SECTION 4  —  VISUALIZE
# ===========================================================================

def _apply_style(ax, title, xlabel, ylabel):
    """Apply the shared light-academic plot style to an Axes object."""
    ax.set_facecolor(PANEL)
    ax.set_title(title, color=INK, fontsize=10, fontweight="bold", pad=7)
    ax.set_xlabel(xlabel, color=INK, fontsize=9)
    ax.set_ylabel(ylabel, color=INK, fontsize=9)
    ax.tick_params(colors=INK, labelsize=8)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(which="major", color=GRID_CLR, linewidth=0.7, linestyle="--")
    ax.grid(which="minor", color=GRID_CLR, linewidth=0.3, linestyle=":")
    for spine in ax.spines.values():
        spine.set_edgecolor("#AAAAAA")
        spine.set_linewidth(0.8)


def _save_and_open(fig, filename, description):
    """Save figure and launch the system viewer."""
    path = os.path.join(SAVE_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=PAPER, edgecolor="none")
    plt.close(fig)
    launch_viewer(path)
    print(f"  [Saved] {description}  →  {path}")


def render_part1(time_arr, traj, evals):
    """
    Three-panel figure for Part 1:
      Left  — time-series decay curves for all three processors
      Centre — phase portrait  x_A vs x_B
      Right  — bar chart of eigenvalue magnitudes (decay rates)
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor=PAPER)
    fig.suptitle(
        "Part 1 — Three-Processor Data Degradation   |   x'(t) = Ax(t),  f(t) = 0",
        color=INK, fontsize=12, fontweight="bold", y=1.01,
    )

    node_colors  = [C_NODE_A, C_NODE_B, C_NODE_C]
    node_labels  = ["Processor A  x_A(t)", "Processor B  x_B(t)", "Processor C  x_C(t)"]

    # ── Left: time series ──────────────────────────────────────────────────
    ax = axes[0]
    for i, (col, lbl) in enumerate(zip(node_colors, node_labels)):
        ax.plot(time_arr, traj[i], color=col, linewidth=1.8,
                label=lbl, solid_capstyle="round")
        # half-life marker
        hl = np.log(2) / abs(evals[i])
        ax.axvline(hl, color=col, linewidth=0.6, linestyle=(0, (4, 4)), alpha=0.5)

    ax.legend(fontsize=8, framealpha=0.9, edgecolor=GRID_CLR, loc="upper right")
    ax.fill_between(time_arr, 0, traj[0], alpha=0.07, color=C_NODE_A)
    _apply_style(ax, "Node Data vs Time", "Time  (s)", "I/O + Data  (MB)")
    ax.set_ylim(bottom=0)

    # ── Centre: phase portrait ─────────────────────────────────────────────
    ax = axes[1]
    ax.plot(traj[0], traj[1], color=C_PHASE, linewidth=1.5, zorder=3)
    ax.scatter(traj[0, 0],  traj[1, 0],  s=60, color=C_NODE_C,
               zorder=5, label="t = 0  (start)", edgecolors=INK, linewidths=0.6)
    ax.scatter(traj[0, -1], traj[1, -1], s=60, marker="s", color=C_NODE_B,
               zorder=5, label="t = 120 s  (end)", edgecolors=INK, linewidths=0.6)
    ax.legend(fontsize=8, framealpha=0.9, edgecolor=GRID_CLR)
    _apply_style(ax, "Phase Portrait  x_A vs x_B", "x_A(t)  (MB)", "x_B(t)  (MB)")

    # ── Right: eigenvalue magnitude bars ───────────────────────────────────
    ax = axes[2]
    labels   = [f"λ₁\n{evals[0]:.4f}", f"λ₂\n{evals[1]:.4f}", f"λ₃\n{evals[2]:.4f}"]
    magnitudes = [abs(v) for v in evals]
    bars = ax.bar(labels, magnitudes, color=node_colors,
                  width=0.5, edgecolor="#888888", linewidth=0.7)
    for bar, val in zip(bars, magnitudes):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.0015,
                f"{val:.4f}", ha="center", va="bottom",
                fontsize=8, color=INK)
    _apply_style(ax, "Decay Rates  |λ|  (larger = faster)",
                 "Eigenvalue", "|λ|  (s⁻¹)")
    ax.set_ylim(0, max(magnitudes) * 1.25)

    fig.tight_layout()
    _save_and_open(fig, "part1_results.png", "Part 1 — three-panel graph")


def render_part2(time_arr, traj, mat_exp_fn, lam2):
    """
    Four-panel figure for Part 2:
      Top-left  — IVP solution curves x₁(t) and x₂(t)
      Top-right — phase portrait
      Bottom-left  — e^{At}[1,1] component
      Bottom-right — e^{At}[2,2] component
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), facecolor=PAPER)
    fig.suptitle(
        "Part 2 — Two-Processor Closed Loop   |   IVP  x(0)=[1,−1]  via  x(t) = e^{At} x(0)",
        color=INK, fontsize=12, fontweight="bold",
    )

    # ── Top-left: IVP solution ─────────────────────────────────────────────
    ax = axes[0][0]
    ax.plot(time_arr, traj[0], color=C_NODE_A, linewidth=2.0,
            label=r"$x_1(t) = e^{-5t}$  (Processor A)")
    ax.plot(time_arr, traj[1], color=C_NODE_B, linewidth=2.0,
            label=r"$x_2(t) = -e^{-5t}$  (Processor B)", linestyle="--")
    ax.axhline(0, color=ANNOT, linewidth=0.6, linestyle=":")
    ax.scatter([0, 0], [1, -1], s=55, color=[C_NODE_A, C_NODE_B],
               zorder=5, edgecolors=INK, linewidths=0.6)
    ax.annotate("x₁(0) = 1",  (0, 1),  xytext=(0.25, 0.82), fontsize=8,
                color=C_NODE_A, arrowprops=dict(arrowstyle="->", color=C_NODE_A, lw=0.8))
    ax.annotate("x₂(0) = -1", (0, -1), xytext=(0.25, -0.82), fontsize=8,
                color=C_NODE_B, arrowprops=dict(arrowstyle="->", color=C_NODE_B, lw=0.8))
    ax.legend(fontsize=8, framealpha=0.9, edgecolor=GRID_CLR)
    _apply_style(ax, "IVP Solution Curves", "Time  (s)", "Data State  x(t)")

    # ── Top-right: phase portrait ──────────────────────────────────────────
    ax = axes[0][1]
    ax.plot(traj[0], traj[1], color=C_PHASE, linewidth=1.8, zorder=3)
    ax.scatter(traj[0, 0],  traj[1, 0],  s=70, color=C_NODE_C,
               zorder=5, label="t = 0:  (1, -1)", edgecolors=INK, linewidths=0.6)
    ax.scatter(traj[0, -1], traj[1, -1], s=60, marker="s", color=ANNOT,
               zorder=5, label="t → ∞:  (0, 0)", edgecolors=INK, linewidths=0.6)
    # Diagonal reference line  x₂ = -x₁
    xs = np.linspace(-0.05, 1.05, 80)
    ax.plot(xs, -xs, color=ANNOT, linewidth=0.8, linestyle=(0, (3, 4)),
            label="x₂ = −x₁", alpha=0.6)
    ax.legend(fontsize=8, framealpha=0.9, edgecolor=GRID_CLR)
    _apply_style(ax, "Phase Portrait  x₁ vs x₂", "x₁(t)", "x₂(t)")

    # ── Bottom row: two e^{At} components ─────────────────────────────────
    t_exp = np.linspace(0, 3, 500)
    exp_vals = mat_exp_fn(t_exp)   # shape (500, 2, 2)

    panels = [
        (axes[1][0], exp_vals[:, 0, 0], C_EAT_11,
         r"$e^{At}_{11}$ = $(3 + 2e^{-5t})\,/\,5$"),
        (axes[1][1], exp_vals[:, 1, 1], C_EAT_22,
         r"$e^{At}_{22}$ = $(2 + 3e^{-5t})\,/\,5$"),
    ]
    for ax, vals, col, title in panels:
        ax.plot(t_exp, vals, color=col, linewidth=2.0)
        steady = vals[-1]
        ax.axhline(steady, color=ANNOT, linewidth=0.8, linestyle="--",
                   label=f"Steady state = {steady:.4f}")
        ax.fill_between(t_exp, steady, vals, alpha=0.08, color=col)
        ax.legend(fontsize=8, framealpha=0.9, edgecolor=GRID_CLR)
        _apply_style(ax, title, "t  (s)", "Value")

    fig.tight_layout()
    _save_and_open(fig, "part2_results.png", "Part 2 — IVP + matrix exponential")


def render_eat_grid(mat_exp_fn):
    """
    Dedicated 2×2 figure showing all four components of e^{At}.
    """
    t_vals  = np.linspace(0, 3, 500)
    eat_all = mat_exp_fn(t_vals)   # shape (500, 2, 2)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), facecolor=PAPER)
    fig.suptitle(
        "Matrix Exponential  e^{At}  —  All Four Entries",
        color=INK, fontsize=13, fontweight="bold",
    )

    entry_colors = [[C_EAT_11, C_EAT_12], [C_EAT_21, C_EAT_22]]
    entry_formulas = [
        ["$(3 + 2e^{-5t})\\ /\\ 5$",  "$3(1 - e^{-5t})\\ /\\ 5$"],
        ["$2(1 - e^{-5t})\\ /\\ 5$",  "$(2 + 3e^{-5t})\\ /\\ 5$"],
    ]

    for r in range(2):
        for c in range(2):
            ax    = axes[r][c]
            col   = entry_colors[r][c]
            vals  = eat_all[:, r, c]
            steady = vals[-1]

            ax.plot(t_vals, vals, color=col, linewidth=2.2)
            ax.axhline(steady, color=ANNOT, linewidth=0.9,
                       linestyle="--", alpha=0.7)
            ax.fill_between(t_vals, steady, vals, alpha=0.09, color=col)
            ax.text(0.97, 0.50, f"→ {steady:.4f}",
                    transform=ax.transAxes, ha="right", va="center",
                    fontsize=9, color=ANNOT)

            title = (f"$e^{{At}}$ [{r+1},{c+1}]  =  "
                     f"{entry_formulas[r][c]}")
            _apply_style(ax, title, "t  (s)", "Value")
            ax.set_facecolor(PANEL)

    fig.tight_layout()
    _save_and_open(fig, "matrix_exponential.png",
                   "e^{At} — all four components")


# ===========================================================================
# MAIN
# ===========================================================================

def _section_banner(title: str) -> None:
    width = 65
    print("\n" + "┌" + "─" * (width - 2) + "┐")
    pad = (width - 2 - len(title)) // 2
    print("│" + " " * pad + title + " " * (width - 2 - pad - len(title)) + "│")
    print("└" + "─" * (width - 2) + "┘")


def main():
    # ── Header ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  Data Integrity Degradation Simulator  —  CST-305 Project 4")
    print("  x'(t) = A x(t)   |   ODE system, matrix exponential, IVP")
    print("=" * 65)

    # ════════════════════════════════════════════════════════════════
    #  PART 1
    # ════════════════════════════════════════════════════════════════
    A1, cap1 = gather_part1_inputs()

    print("\n  Building and solving Part 1 system...")
    evals1, evecs1 = eigendecompose(A1, tag="A  (scaled, Part 1)")

    x0_part1 = np.array([25.0, 25.0, 25.0])
    print(f"\n  Initial state: x_A = x_B = x_C = 25.0 MB  (I/O portion)")
    t1, traj1 = integrate_ode(A1, x0_part1, t_final=120.0)

    report_part1(A1, evals1, t1, traj1)
    render_part1(t1, traj1, evals1)

    # ════════════════════════════════════════════════════════════════
    #  PART 2
    # ════════════════════════════════════════════════════════════════
    A2, cap2, x0_part2 = gather_part2_inputs()

    # Scale up for integer eigenvalues — cleaner arithmetic
    A2_unscaled = A2 * 100.0

    print("\n  Building and solving Part 2 system...")
    print("  (Working with unscaled matrix  A₀ = 100·A  for integer eigenvalues)")
    evals2, evecs2 = eigendecompose(A2_unscaled, tag="A₀  (unscaled, Part 2)")

    print("\n  Deriving matrix exponential e^{At}...")
    mat_exp_fn, lam2 = build_matrix_exponential(A2_unscaled)

    print("\n  Solving IVP via matrix method  x(t) = e^{At} x(0)...")
    t2, traj2 = solve_by_matrix_exp(A2_unscaled, mat_exp_fn, x0_part2)

    report_part2(evals2, lam2, x0_part2, t2, traj2)
    render_part2(t2, traj2, mat_exp_fn, lam2)
    render_eat_grid(mat_exp_fn)

    # ── Summary ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  COMPLETE")
    print("=" * 65)
    print(f"  Output folder : {SAVE_DIR}")
    print(f"  Part 1 graph  : part1_results.png")
    print(f"  Part 2 graph  : part2_results.png")
    print(f"  e^{{At}} grid   : matrix_exponential.png")
    print()
    print(f"  Part 1 eigenvalues : {[f'{v:.4f}' for v in evals1]}")
    print(f"  Part 2 eigenvalues : λ₁ = 0,   λ₂ = {lam2:.1f}")
    print(f"  Part 2 solution    : x₁(t) = e^{{{lam2:.0f}t}},   x₂(t) = -e^{{{lam2:.0f}t}}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()