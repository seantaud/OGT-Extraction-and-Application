import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np
from matplotlib import image as mpimg


class SeabornFig2Grid():

    def __init__(self, seaborngrid, fig, subplot_spec):
        self.fig = fig
        self.sg = seaborngrid
        self.subplot = subplot_spec
        if isinstance(self.sg, sns.axisgrid.FacetGrid) or \
                isinstance(self.sg, sns.axisgrid.PairGrid):
            self._movegrid()
        elif isinstance(self.sg, sns.axisgrid.JointGrid):
            self._movejointgrid()
        self._finalize()

    def _movegrid(self):
        """ Move PairGrid or Facetgrid """
        self._resize()
        n = self.sg.axes.shape[0]
        m = self.sg.axes.shape[1]
        self.subgrid = gridspec.GridSpecFromSubplotSpec(n, m, subplot_spec=self.subplot)
        for i in range(n):
            for j in range(m):
                self._moveaxes(self.sg.axes[i, j], self.subgrid[i, j])

    def _movejointgrid(self):
        """ Move Jointgrid """
        h = self.sg.ax_joint.get_position().height
        h2 = self.sg.ax_marg_x.get_position().height
        r = int(np.round(h / h2))
        self._resize()
        self.subgrid = gridspec.GridSpecFromSubplotSpec(r + 1, r + 1, subplot_spec=self.subplot)

        self._moveaxes(self.sg.ax_joint, self.subgrid[1:, :-1])
        self._moveaxes(self.sg.ax_marg_x, self.subgrid[0, :-1])
        self._moveaxes(self.sg.ax_marg_y, self.subgrid[1:, -1])

    def _moveaxes(self, ax, gs):
        # https://stackoverflow.com/a/46906599/4124317
        ax.remove()
        ax.figure = self.fig
        self.fig.axes.append(ax)
        self.fig.add_axes(ax)
        ax._subplotspec = gs
        ax.set_position(gs.get_position(self.fig))
        ax.set_subplotspec(gs)

    def _finalize(self):
        plt.close(self.sg.fig)
        self.fig.canvas.mpl_connect("resize_event", self._resize)
        self.fig.canvas.draw()

    def _resize(self, evt=None):
        self.sg.fig.set_size_inches(self.fig.get_size_inches())


def set_plot(font_scale=2):
    plt.rcParams['figure.dpi'] = 75
    sns.set_theme(style='whitegrid')
    sns.set_style('whitegrid', {'font.sans-serif': ['simhei', 'Arial']})
    plt.rcParams['axes.unicode_minus'] = False
    sns.set(font_scale=font_scale)


def get_color(pos):
    palette=sns.color_palette('Blues', n_colors=4),
    from matplotlib.colors import rgb2hex

    color = [rgb2hex(i) for i in palette[0]]
    color = color[1:]
    return color[pos]


def draw_grid(g0, g1, g2, g3, dst_img):
    fig = plt.figure(figsize=(25, 25))
    # fig.subplots_adjust(hspace=0.825, wspace=0.825)
    gs = gridspec.GridSpec(2, 2)

    mg0 = SeabornFig2Grid(g0, fig, gs[0])
    mg1 = SeabornFig2Grid(g1, fig, gs[1])
    mg2 = SeabornFig2Grid(g2, fig, gs[2])
    mg3 = SeabornFig2Grid(g3, fig, gs[3])

    # gs.tight_layout(fig)
    # fig.savefig(dst_img + "/4.png", format='png', dpi=100, bbox_inches='tight')
    # plt.show()

    ############### 2. SAVE PLOTS IN MEMORY TEMPORALLY
    g0.savefig(dst_img + 'g0.png',format='png', dpi=100, bbox_inches='tight')
    plt.close(g0.fig)

    g1.savefig(dst_img + 'g1.png',format='png', dpi=100, bbox_inches='tight')
    plt.close(g1.fig)

    g2.savefig(dst_img + 'g2.png',format='png', dpi=100, bbox_inches='tight')
    plt.close(g2.fig)

    g3.savefig(dst_img + 'g3.png',format='png', dpi=100, bbox_inches='tight')
    plt.close(g3.fig)

    ############### 3. CREATE YOUR SUBPLOTS FROM TEMPORAL IMAGES
    f, axarr = plt.subplots(2, 2, figsize=(40, 40))

    axarr[0, 0].imshow(mpimg.imread(dst_img + 'g0.png'))
    axarr[0, 1].imshow(mpimg.imread(dst_img + 'g1.png'))
    axarr[1, 0].imshow(mpimg.imread(dst_img + 'g3.png'))
    axarr[1, 1].imshow(mpimg.imread(dst_img + 'g2.png'))

    # turn off x and y axis
    [ax.set_axis_off() for ax in axarr.ravel()]

    # plt.tight_layout()
    plt.savefig(dst_img + "4.png", format='png', dpi=100, bbox_inches='tight')
    plt.show()
    f,axarr = plt.subplots(1, 1, figsize=(25, 25))
    axarr.imshow(mpimg.imread(dst_img + '4.png'))
    plt.show()





def histplot_dif(x, color, **kwargs):
    cmap = sns.light_palette(color, as_cmap=True)
    sns.histplot(x, cmap=cmap, **kwargs)
