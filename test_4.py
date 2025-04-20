import numpy as np
import OFDM_STEEP_2 as mos
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == "__main__":
    colors = sns.color_palette("muted")[:4]
    sns.set_style("whitegrid")

    SABs = [10, 20]
    SBAs = np.linspace(-15, 15, 16)
    alpha = 3
    beta = 5
    N = 50
    nE = 4

    SAB = SABs[0]
    p2 = np.zeros(len(SBAs))
    p4 = np.zeros(len(SBAs))
    p5 = np.zeros(len(SBAs))
    for i in range(len(SBAs)):
        print("\n", i + 1, " of ", len(SBAs), "\n")
        p2[i], p4[i], p5[i] = mos.get_rate_upper_both(
            mos.in_db(SAB), mos.in_db(SBAs[i]), alpha, beta, N, nE
        )
    plt.plot(
        SBAs,
        p2,
        linestyle="-",
        color=colors[0],
        marker="o",
        label="policy-2; $S_{AB}=$" + f"{SAB} dB",
    )
    plt.plot(
        SBAs,
        p4,
        linestyle="-",
        color=colors[1],
        marker="*",
        label="policy-4; $S_{AB}=$" + f"{SAB} dB",
    )
    plt.plot(
        SBAs,
        p5,
        linestyle="-",
        color=colors[2],
        marker="d",
        label="policy-5; $S_{AB}=$" + f"{SAB} dB",
    )

    SAB = SABs[1]
    p2 = np.zeros(len(SBAs))
    p4 = np.zeros(len(SBAs))
    p5 = np.zeros(len(SBAs))
    for i in range(len(SBAs)):
        print("\n", i + 1, " of ", len(SBAs), "\n")
        p2[i], p4[i], p5[i] = mos.get_rate_upper_both(
            mos.in_db(SAB), mos.in_db(SBAs[i]), alpha, beta, N, nE
        )
    plt.plot(
        SBAs,
        p2,
        linestyle="--",
        color=colors[0],
        marker="o",
        label="policy-2; $S_{AB}=$" + f"{SAB} dB",
    )
    plt.plot(
        SBAs,
        p4,
        linestyle="--",
        color=colors[1],
        marker="*",
        label="policy-4; $S_{AB}=$" + f"{SAB} dB",
    )
    plt.plot(
        SBAs,
        p5,
        linestyle="--",
        color=colors[2],
        marker="d",
        label="policy-5; $S_{AB}=$" + f"{SAB} dB",
    )

    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.xlabel("$S_{BA}$ in dB", fontsize=12)
    plt.ylabel("avg secrecy in single carrier", fontsize=12)
    plt.title(
        f"$n_E={nE}$" + f"; $N={N}$" + f"; $\\alpha={alpha}$" + f"; $\\beta={beta}$"
    )
    plt.savefig("pplot4.png", bbox_inches="tight", dpi=300)
    plt.show()
