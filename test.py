import numpy as np
import mpmath as mp

# test.py
# Plots values of the Riemann zeta function on the critical line and attempts
# to locate nontrivial zeros (visual illustration; not a proof of the Riemann Hypothesis).
#
# Requires: mpmath, numpy, matplotlib
# Install with: pip install mpmath numpy matplotlib

import matplotlib.pyplot as plt

mp.mp.dps = 50  # working precision (increase if you need more accuracy)


def find_zeros_on_critical_line(t_min=5, t_max=200, seeds=400):
    """
    Try to locate zeros of zeta(s) with s = 0.5 + i t by using many starting seeds
    for mpmath.findroot. Returns a sorted list of complex roots (unique by tolerance).
    """
    seeds_t = np.linspace(t_min, t_max, seeds)
    found = []
    tol_im = 1e-8
    tol_re = 1e-8

    for t0 in seeds_t:
        s0 = mp.mpc(0.5, t0)
        try:
            root = mp.findroot(lambda z: mp.zeta(z), s0, tol=1e-30, maxsteps=50)
            r_re = float(mp.nstr(root.real, 12))
            r_im = float(mp.nstr(root.imag, 12))
            candidate = complex(r_re, r_im)
            # deduplicate: consider close roots equal
            if not any(abs(candidate.real - r.real) < tol_re and abs(candidate.imag - r.imag) < tol_im for r in found):
                found.append(candidate)
        except Exception:
            # many seeds will not converge; ignore failures
            pass

    found.sort(key=lambda z: z.imag)
    return found


def zeta_on_critical(t_vals):
    """Return |zeta(0.5 + i t)| for array of t values"""
    return np.array([abs(mp.zeta(mp.mpc(0.5, float(t)))) for t in t_vals], dtype=float)


def plot_results(zeros, t_min=0.0, t_max=200.0, t_steps=4000):
    t = np.linspace(t_min, t_max, t_steps)
    abs_vals = zeta_on_critical(t)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9))

    # Plot |zeta(0.5 + i t)| vs t and mark found zeros
    ax1.plot(t, abs_vals, lw=1, label=r'|ζ(1/2 + i t)|')
    if zeros:
        zeros_t = [z.imag for z in zeros]
        zeros_abs = [0.0 for _ in zeros]
        ax1.scatter(zeros_t, zeros_abs, color='red', s=30, label='found zeros (imag part)')
        for z in zeros:
            ax1.axvline(z.imag, color='red', alpha=0.2, lw=0.7)
    ax1.set_xlim(t_min, t_max)
    ax1.set_ylim(0, np.percentile(abs_vals, 98)*1.2)
    ax1.set_xlabel('t')
    ax1.set_ylabel('|ζ(1/2 + i t)|')
    ax1.set_title('Magnitude of ζ on the critical line and located zeros')
    ax1.legend()

    # Scatter plot of zeros in the critical strip (Re in [0,1]) for context
    ax2.set_title('Located zeros (critical strip view)')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(t_min, t_max)
    ax2.set_xlabel('Re(s)')
    ax2.set_ylabel('Im(s) (t)')
    if zeros:
        ax2.scatter([z.real for z in zeros], [z.imag for z in zeros], color='red')
    # also draw the critical line Re=0.5
    ax2.axvline(0.5, color='black', linestyle='--', lw=1, label='critical line Re(s)=1/2')
    ax2.legend()

    plt.tight_layout()
    plt.show()


def main():
    # parameters - adjust as desired
    t_min = 5
    t_max = 200
    seed_count = 600

    print("Searching for zeros on the critical line between t = {:.1f} and {:.1f}...".format(t_min, t_max))
    zeros = find_zeros_on_critical_line(t_min=t_min, t_max=t_max, seeds=seed_count)
    print("Found {} candidate zeros (numerical approximations):".format(len(zeros)))
    for z in zeros:
        print("  s ≈ {:.12f} + {:.12f} i".format(z.real, z.imag))

    plot_results(zeros, t_min=0.0, t_max=t_max, t_steps=4000)


if __name__ == "__main__":
    main()
    