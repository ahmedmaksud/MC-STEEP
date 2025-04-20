import numpy as np
import OFDM_STEEP_2 as mos
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == "__main__":
    colors = sns.color_palette("muted")[:4]
    sns.set_style("whitegrid")

    SAB = 20
    SBAs = np.linspace(-10, 20, 16)
    alpha = 3
    beta = 5
    Ns = [50, 100]
    nE = 4

    N = Ns[0]
    p1 = np.zeros(len(SBAs))
    p2 = np.zeros(len(SBAs))
    p3 = np.zeros(len(SBAs))
    for i in range(len(SBAs)):
        print(i + 1, " of ", len(SBAs))
        p1[i], p2[i], p3[i] = mos.get_rate_upper_nopower(
            mos.in_db(SAB), mos.in_db(SBAs[i]), alpha, beta, N, nE
        )
    plt.plot(
        SBAs,
        p1,
        linestyle="-",
        color=colors[0],
        marker="o",
        label="policy-1;$N=$" + f"{N}",
    )
    plt.plot(
        SBAs,
        p2,
        linestyle="-",
        color=colors[1],
        marker="*",
        label="policy-2;$N=$" + f"{N}",
    )
    plt.plot(
        SBAs,
        p3,
        linestyle="-",
        color=colors[2],
        marker="^",
        label="policy-3;$N=$" + f"{N}",
    )

    N = Ns[1]
    p1 = np.zeros(len(SBAs))
    p2 = np.zeros(len(SBAs))
    p3 = np.zeros(len(SBAs))
    for i in range(len(SBAs)):
        print(i + 1, " of ", len(SBAs))
        p1[i], p2[i], p3[i] = mos.get_rate_upper_nopower(
            mos.in_db(SAB), mos.in_db(SBAs[i]), alpha, beta, N, nE
        )
    plt.plot(
        SBAs,
        p1,
        linestyle="--",
        color=colors[0],
        marker="o",
        label="policy-1;$N=$" + f"{N}",
    )
    plt.plot(
        SBAs,
        p2,
        linestyle="--",
        color=colors[1],
        marker="*",
        label="policy-2;$N=$" + f"{N}",
    )
    plt.plot(
        SBAs,
        p3,
        linestyle="--",
        color=colors[2],
        marker="^",
        label="policy-3;$N=$" + f"{N}",
    )

    # N = Ns[2]
    # p1 = np.zeros(len(SBAs))
    # p2 = np.zeros(len(SBAs))
    # p3 = np.zeros(len(SBAs))
    # for i in range(len(SBAs)):
    #     print(i+1, ' of ', len(SBAs))
    #     p1[i], p2[i], p3[i] = mos.get_rate_upper_nopower(mos.in_db(SAB),
    #                                                      mos.in_db(SBAs[i]),
    #                                                      alpha, beta, N, nE)
    # plt.plot(SBAs, p1, linestyle=':', color=colors[0], marker='o',
    #          label='policy-1;$N=$'+f'{N}')
    # plt.plot(SBAs, p2, linestyle=':', color=colors[1], marker='*',
    #          label='policy-2;$N=$'+f'{N}')
    # plt.plot(SBAs, p3, linestyle=':', color=colors[2], marker='^',
    #          label='policy-3;$N=$'+f'{N}')

    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.xlabel("$S_{BA}$ in dB", fontsize=12)
    plt.ylabel("avg secrecy in single carrier", fontsize=12)
    plt.title(
        f"$n_E={nE}$"
        + "; $S_{AB}=$"
        + f"{SAB} dB"
        + f"; $\\alpha={alpha}$"
        + f"; $\\beta={beta}$"
    )
    plt.savefig("pplot2.png", bbox_inches="tight", dpi=300)
    plt.show()
