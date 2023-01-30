#!/usr/bin/self python3
# -*- coding: utf-8 -*-
"""Provides the Visualization class, for rendering episodes."""
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable


class Visualization:
    """ A class for visualizing the search in 1D, 2D or 3D

    Args:
        env (SourceTracking):
            an instance of the SourceTracking class
        live (bool, optional):
            whether to show live preview (faster if False) (default=False)
        filename (str, optional):
            file name for the video (default='test')
        log_prob (bool, optional):
            whether to show log(prob) instead of prob (default=False)
        marginal_prob_3d (bool, optional):
            in 3D, whether to show marginal pdfs on each plane, instead of the pdf in the planes that the
            agent crosses (default=False)
    """
    
    def __init__(self,
                 env,
                 live=False,
                 filename='test',
                 log_prob=False,
                 marginal_prob_3d=False,
                 ):
        self.env = env
        if self.env.Ndim != 2:
            raise Exception("Problem with Ndim: visualization is not possible")

        self.video_live = live
        self.frame_path = filename + "_frames"
        self.video_path = filename + "_video"
        if not os.path.isdir(self.frame_path):
            os.mkdir(self.frame_path)

        self.log_prob = log_prob
        self.marginal_prob_3d = marginal_prob_3d

    def make_video(self, frame_rate=5, keep_frames=False):
        """
        Make a video from recorded frames and clean up frames.

        Args:
            frame_rate (int): number of frames per second (default=5)
            keep_frames (bool): whether to keep the frames as images (default=False)

        Returns:
            exit_code (int):
                nonzero if something went wrong while making the video, in that case frames will be
                saved even if keep_frames = False
        """
        if self.video_live:
            plt.close("all")

        exit_code = self._make_video(frame_rate=frame_rate, keep_frames=keep_frames)
        return exit_code

    def record_snapshot(self, num, toptext=''):
        """Create a frame from current state of the search, and save it.

        Args:
            num (int): frame number (used to create filename)
            toptext (str): text that will appear in the top part of the frame (like a title)
        """

        if self.video_live:
            if not hasattr(self, 'fig'):
                fig, ax = self._setup_render()
                ax[0].set_title("observation map (current: %s)" % self._obs_to_str())
                ax[1].set_title("source probability distribution (entropy = %.3f)" % self.env.entropy)
                self.fig = fig
                self.ax = ax
            else:
                fig = self.fig
                ax = self.ax
                ax[0].title.set_text("observation map (current: %s)" % self._obs_to_str())
                ax[1].title.set_text("source probability distribution (entropy = %.3f)" % self.env.entropy)
        else:
            fig, ax = self._setup_render()
            ax[0].set_title("observation map (current: %s)" % self._obs_to_str())
            ax[1].set_title("source probability distribution (entropy = %.3f)" % self.env.entropy)

        self._update_render(fig, ax, toptext=toptext)

        if self.video_live:
            plt.pause(0.1)
        plt.draw()
        framefilename = self._framefilename(num)
        fig.savefig(framefilename, dpi=150)
        if not self.video_live:
            plt.close(fig)

    # ________internal___________________________________________________________________
    def _obs_to_str(self, ):
        if self.env.obs["done"]:
            out = "source found"
        else:
            if self.env.draw_source:
                out = "source not found, hit = " + str(self.env.obs["hit"])
            else:
                out = "hit = " + str(self.env.obs["hit"])
        return out

    def _setup_render(self, ):

        figsize = (8, 8)

        if self.env.Ndim == 2:
            fig, ax = plt.subplots(2, 1, figsize=figsize)

            # setup figure
            bottom = 0.08
            top = 0.9
            left = 0.02
            right = 0.96
            plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, hspace=0.25)

            # state
            cmap0 = self._cmap0()
            sm0 = plt.cm.ScalarMappable(norm=colors.Normalize(vmin=-0.5, vmax=self.env.Nhits - 0.5), cmap=cmap0)
            divider = make_axes_locatable(ax[0])
            cax0 = divider.append_axes("right", size="5%", pad=0.3)
            fig.colorbar(sm0, cax=cax0, ticks=np.arange(0, self.env.Nhits))
            ax[0].set_aspect("equal", adjustable="box")
            ax[0].axis("off")

            # p_source
            cmap1 = self._cmap1()
            if self.log_prob:
                sm1 = plt.cm.ScalarMappable(norm=colors.LogNorm(vmin=1e-3, vmax=1.0), cmap=cmap1)
            else:
                sm1 = plt.cm.ScalarMappable(norm=colors.Normalize(vmin=np.min(self.env.p_source), vmax=np.max(self.env.p_source)), cmap=cmap1)
            divider = make_axes_locatable(ax[1])
            cax1 = divider.append_axes("right", size="5%", pad=0.3)
            if self.log_prob:
                cbar1 = fig.colorbar(sm1, cax=cax1, extend="min")
            else:
                cbar1 = fig.colorbar(sm1, cax=cax1)
            if self.video_live:
                self.cbar1 = cbar1
            ax[1].set_aspect("equal", adjustable="box")
            ax[1].axis("off")

            # position of source
            if self.env.draw_source:
                for i in range(2):
                    ax[i].plot(self.env.source[0], self.env.source[1], color="r", marker="$+$", markersize=8, zorder=10000)

        return fig, ax

    def _update_render(self, fig, ax, toptext=''):

        if self.video_live:
            if hasattr(self, 'artists'):
                for artist in range(len(self.artists)):
                    if self.artists[artist] is not None:
                        if isinstance(self.artists[artist], list):
                            for art in self.artists[artist]:
                                art.remove()
                        else:
                            self.artists[artist].remove()

        if self.env.Ndim == 1:
            self._draw_1D(fig, ax)
        elif self.env.Ndim == 2:
            self._draw_2D(fig, ax)
        elif self.env.Ndim == 3:
            self._draw_3D(fig, ax)

        bottomtext = "$\\bar{R} = $" + str(self.env.R_bar) \
                     + "$\qquad$ $\\bar{V} = $" + str(self.env.V_bar) \
                     + "$\qquad$ $\\bar{\\tau} = $" + str(self.env.tau_bar) \
                     + "$\qquad$ $\\bar{\lambda} = $" + str(np.round(self.env.lambda_bar, 2)) \
                     + "$\qquad$ $h_{\mathrm{init}}$ = " + str(self.env.initial_hit)
        sup = plt.figtext(0.5, 0.99, toptext, fontsize=12, ha="center", va="top")
        bot = plt.figtext(0.5, 0.01, bottomtext, fontsize=10, ha="center", va="bottom")
        if self.video_live:
            self.artists.append(sup)
            self.artists.append(bot)

    def _draw_1D(self, fig, ax):
        pass

    def _draw_2D(self, fig, ax):
        # hit map
        cmap0 = self._cmap0()
        img0 = ax[0].imshow(
            np.transpose(self.env.hit_map),
            vmin=-0.5,
            vmax=self.env.Nhits - 0.5,
            origin="lower",
            cmap=cmap0,
        )

        # p_source
        cmap1 = self._cmap1()
        img1 = ax[1].imshow(
            np.transpose(self.env.p_source),
            vmin=np.min(self.env.p_source),
            vmax=np.max(self.env.p_source),
            origin="lower",
            aspect='equal',
            cmap=cmap1,
        )
        if self.video_live:
            if self.log_prob:
                sm1 = plt.cm.ScalarMappable(norm=colors.LogNorm(vmin=1e-3, vmax=1.0), cmap=cmap1)
            else:
                sm1 = plt.cm.ScalarMappable(norm=colors.Normalize(vmin=np.min(self.env.p_source), vmax=np.max(self.env.p_source)), cmap=cmap1)
            self.cbar1.update_normal(sm1)

        # position of agent
        aloc = [0] * 2
        for i in range(2):
            aloc[i] = ax[i].plot(self.env.agent[0], self.env.agent[1], "ro")

        if self.video_live:
            self.artists = [img0, img1] + [a for a in aloc]

    def _draw_3D(self, fig, ax):
        pass

    def _cmap0(self):
        topcolors = plt.cm.get_cmap('Greys', 128)
        if self.env.Ndim == 3:
            bottomcolors = plt.cm.get_cmap('jet', 128)
        else:
            if self.env.Nhits > 2:
                bottomcolors = plt.cm.get_cmap('Spectral_r', 128)
                newcolors = np.vstack((topcolors(0.5),
                                       bottomcolors(np.linspace(0, 1, self.env.Nhits - 1))))
            else:
                bottomcolors = plt.cm.get_cmap('Oranges', 128)
                newcolors = np.vstack((topcolors(0.5),
                                       bottomcolors(0.5)))

        cmap0 = ListedColormap(newcolors, name='GreyColors')
        if self.env.Ndim == 2:
            cmap0.set_under(color="black")
        return cmap0

    def _cmap1(self):
        if self.env.Ndim == 1:
            cmap1 = plt.cm.get_cmap("jet", 50)
        elif self.env.Ndim == 2:
            cmap1 = plt.cm.get_cmap("viridis", 50)
        elif self.env.Ndim == 3:
            cmap1 = plt.cm.get_cmap("Blues", 50)
        return cmap1

    def _alpha0(self):
        alpha0 = None
        if self.env.Ndim == 3:
            alpha0 = 0.7
        return alpha0

    def _alpha1(self):
        alpha1 = None
        if self.env.Ndim == 3:
            alpha1 = 0.7
        return alpha1

    def _framefilename(self, num):
        framefilename = os.path.join(self.frame_path,
                                     str(os.path.basename(self.video_path) + "_" + str(num).zfill(8) + ".png"))
        return framefilename

    def _make_video(self, frame_rate, keep_frames):
        out = self.video_path + ".mp4"
        cmd = "ffmpeg -loglevel quiet -r " + str(frame_rate) + " -pattern_type glob -i '" + \
              os.path.join(self.frame_path, "*.png'") + " -c:v libx264 " + out
        exit_code = os.system(cmd)
        if exit_code != 0:
            print("Warning: could not make a video, is ffmpeg installed?")
        else:
            if not keep_frames:
                shutil.rmtree(self.frame_path)
        return exit_code
