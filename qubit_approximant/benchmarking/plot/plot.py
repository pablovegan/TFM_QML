def box_plot_errores(layer_list, cost_array, function):
    # plot
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].boxplot(
        cost_array,
        meanline=True,
        patch_artist=True,
        medianprops={"color": "crimson", "linewidth": 1},
        boxprops={"facecolor": "lightsteelblue", "edgecolor": "black", "linewidth": 0.5},
    )
    """, positions=layer_list, widths=1.5, patch_artist=True,
                showmeans=False, showfliers=False,
                medianprops={"color": "white", "linewidth": 0.5},
                boxprops={"facecolor": "lightsteelblue", "edgecolor": "white", "linewidth": 0.5},
                whiskerprops={"color": "C0", "linewidth": 1.5})"""
    ax[0].set_title(function)
    ax[0].set_xlabel("Capas")
    ax[0].set_ylabel("Error")
    ax[0].set_yscale("log")

    violin = ax[1].violinplot(cost_array, showmeans=True, showmedians=True)
    ax[1].set_title(function)
    ax[1].set_xlabel("Capas")
    ax[1].set_xticks([y + 1 for y in range(cost_array.shape[1])])
    ax[1].set_ylabel("Error")
    ax[1].set_yscale("log")

    violin["cmedians"].set_color("crimson")

    plt.show()
