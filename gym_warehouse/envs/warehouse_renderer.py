"""
Renderer for the warehouse.
"""
import glob
import os
import pathlib
from os.path import isfile, join

import cv2
import numpy as np
import pygame
from PIL import Image

# Constants
WHITE = (255, 255, 255)
DARK_GREY = (64, 64, 64)
BLACK = (0, 0, 0)
GREY = (160, 160, 160)
BLUE = (51, 51, 255)
DARK_GREEN = (64, 140, 85)
DARK_ORANGE = (210, 140, 0)

RED = (255, 0, 0)
DARK_RED = (156, 0, 0)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
DARK_YELLOW = (255, 255, 142)
TEAL = (0, 255, 255)
PINK = (255, 0, 255)

ITEM_COLORS = [WHITE, RED, GREEN, YELLOW, PINK, TEAL]

# Colors similar to pygame, following colormap gist_ncar, on a 30-color scale
C_MAX = 30
C_LEGEND = 0
C_BIN = 2
C_AGENT = 1
C_SPOT = 28
C_FLOOR = 29
C_ITEMS = [6, 22, 11, 16, 24, 8]

CELL_SIZE = 50
CELL_MARGIN = 5


class WarehouseRenderer:
    """
    A renderer for the warehouse using pygame.

    Parameters
    ----------
    env : WarehouseEnv
        The warehouse environment to render.
    """

    def __init__(self, env):
        pygame.display.init()
        pygame.font.init()
        self.env = env
        self.width = env.num_cols * CELL_SIZE
        self.height = (env.num_rows + 1) * CELL_SIZE
        # self.screen = pygame.display.set_mode((self.width, self.height))
        hh = (8 * 650) // 7
        self.screen = pygame.display.set_mode((hh, 400))
        self.num_slots = max(self.env.num_bin_slots, self.env.num_agent_slots)
        self.slot_height = (CELL_SIZE - 2 * CELL_MARGIN) // self.num_slots
        self.slot_width = CELL_SIZE - 2 * CELL_MARGIN
        pygame.display.set_caption("Chaotic Warehouse")

        self.state = np.array([])
        self.state_history = []

        pathlib.Path("./Surface").mkdir(exist_ok=True)
        old_frames = "./Surface/surface_*.PNG"
        for frame in glob.glob(old_frames):
            os.remove(frame)
        self.surface_index = 0

    def render(self):
        """
        Renders the warehouse to the screen.
        """
        self.screen.fill(WHITE)
        self._draw_warehouse_object(self.env.staging_in_area, DARK_GREEN)
        self._draw_warehouse_object(self.env.staging_out_area, DARK_ORANGE)

        for bin_ in self.env.bins:
            self._draw_warehouse_object(bin_, GREY)

        for obst in self.env.obstacles_only:
            self._draw_warehouse_object(obst, BLACK)

        self._draw_warehouse_object(self.env.agent, BLUE)
        self._draw_panel()

        pygame.display.update()

        # Save surfaces as images
        surface = pygame.display.get_surface()
        pygame.image.save(
            surface, "Surface/surface_" + str(self.surface_index) + ".PNG"
        )
        self.surface_index += 1

    def save_as_gif(self, fp_out="./Surface/animation.gif", sec_per_frame=0.5):
        """
        Save Interaction between agent and wh as an animation in gif format
        """

        fp_in = "./Surface/surface_*.PNG"
        imgs = []
        file_number = []

        for frame in glob.glob(fp_in):
            img = Image.open(frame)
            imgs.append(img)
            file_number.append(img.filename[18:-4])

        index = np.asarray(file_number).astype(int).argsort()
        ordered_imgs = [imgs[i] for i in index]

        duration = sec_per_frame * 1000  # Every frame will run for "duration"

        img.save(
            fp=fp_out,
            format="GIF",
            append_images=ordered_imgs,
            save_all=True,
            duration=duration,
            optimize=False,
        )

    def save_as_mp4(self, fp_out="./video.pm4", sec_per_frame=0.5):
        """
        Save Interaction between agent and wh as a video in mp4 format
        """

        path_in = "./Surface/"
        fps = 1 // sec_per_frame

        files = []
        imgs = []
        file_number = []

        for f in os.listdir(path_in):
            if isfile(join(path_in, f)) and ".PNG" in f:
                files.append(f)
                file_number.append(f[8:-4])

        # order correctly the files
        index = np.asarray(file_number).astype(int).argsort()
        files = [files[i] for i in index]

        for i in range(len(files)):
            file_path = path_in + files[i]
            img = cv2.imread(file_path)
            height, width, layers = img.shape
            size = (width, height)

            # inserting the frames into an image array
            imgs.append(img)

        out = cv2.VideoWriter(fp_out, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

        for i in range(len(imgs)):
            # writing to a image array
            out.write(imgs[i])
        out.release()

    def _draw_warehouse_object(self, warehouse_object, color, video=False):
        """
        Draws the warehouse object using the given color.
        Also draws the contents, and if applicable, the access spots of the
        object.
        """
        if not video:
            r, c = warehouse_object.position
            y, x = r * CELL_SIZE, c * CELL_SIZE
            rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(self.screen, color, rect, 0)

            x += CELL_MARGIN
            y += CELL_MARGIN
            for item in warehouse_object.status:
                if item != 0:
                    rect = pygame.Rect(x, y, self.slot_width, self.slot_height)
                    pygame.draw.rect(self.screen, ITEM_COLORS[item], rect, 0)
                y += self.slot_height

            if hasattr(warehouse_object, "access_spots"):
                for access_spot in warehouse_object.access_spots:
                    r, c = access_spot
                    y, x = r * CELL_SIZE, c * CELL_SIZE
                    rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
                    pygame.draw.rect(self.screen, DARK_YELLOW, rect, 0)
        else:
            r, c = warehouse_object.position
            y, x = r * self.num_slots, c * self.num_slots
            self.state[y : y + self.num_slots, x : x + self.num_slots] = color / C_MAX

            for item in warehouse_object.status:
                if item != 0:
                    self.state[y, x : x + self.num_slots] = C_ITEMS[item] / C_MAX
                y += 1

            if hasattr(warehouse_object, "access_spots"):
                for access_spot in warehouse_object.access_spots:
                    r, c = access_spot
                    y, x = r * self.num_slots, c * self.num_slots
                    self.state[y : y + self.num_slots, x : x + self.num_slots] = (
                        C_SPOT / C_MAX
                    )

    def _draw_panel(self):
        """
        Draws a panel at the bottom of the screen displaying the active
        transactions and the current reward.
        """
        y = self.height - CELL_SIZE
        rect = pygame.Rect(0, y, self.width, CELL_SIZE)
        pygame.draw.rect(self.screen, DARK_GREY, rect, 0)
        font = pygame.font.SysFont("Arial", 12)
        text_surface = font.render("Inbound:", True, WHITE)
        self.screen.blit(text_surface, (0, y + CELL_SIZE // 2))
        self._draw_transaction(CELL_SIZE, y, self.env.staging_in_area)
        text_surface = font.render("Outbound:", True, WHITE)
        self.screen.blit(text_surface, (2 * CELL_SIZE, y + CELL_SIZE // 2))
        self._draw_transaction(3 * CELL_SIZE, y, self.env.staging_out_area)
        text_surface = font.render(f"a = {self.env.action}", True, WHITE)
        self.screen.blit(text_surface, (4 * CELL_SIZE, y + CELL_SIZE // 4))
        text_surface = font.render(f"r = {self.env.reward}", True, WHITE)
        self.screen.blit(text_surface, (4 * CELL_SIZE, y + 2 * CELL_SIZE // 4))
        text_surface = font.render(
            f"t = {self.env.time_step}/{self.env.time_limit}", True, WHITE
        )
        self.screen.blit(text_surface, (4 * CELL_SIZE, y + 3 * CELL_SIZE // 4))

    def _draw_transaction(self, x, y, staging_area, video=False):
        """
        Draws the transaction of the given staging area in the cell with given
        top left corner coordinates.
        """
        if not video:
            x += CELL_MARGIN
            y += CELL_MARGIN
            for item in staging_area.status:
                if item != 0:
                    rect = pygame.Rect(x, y, self.slot_width, self.slot_height)
                    pygame.draw.rect(self.screen, ITEM_COLORS[item], rect, 0)
                y += self.slot_height
        else:
            for item in staging_area.status:
                if item != 0:
                    self.state[y, x : x + self.num_slots] = C_ITEMS[item] / C_MAX
                y += 1

    def save_state(self):
        """
        Convert the current environment state into a grid that can be plotted
        afterwards during the movie generation. Save it in self.state_history.
        """
        self.state = (
            np.ones(
                (
                    (self.env.num_rows + 2) * self.num_slots,
                    self.env.num_cols * self.num_slots,
                )
            )
            * C_FLOOR
            / C_MAX
        )

        self._draw_warehouse_object(self.env.staging_in_area, C_BIN, True)
        self._draw_warehouse_object(self.env.staging_out_area, C_BIN, True)

        for bin_ in self.env.bins:
            self._draw_warehouse_object(bin_, C_BIN, True)

        self._draw_warehouse_object(self.env.agent, C_AGENT, True)

        # Plot transactions
        y, x = (self.env.num_rows * self.num_slots, self.env.num_cols * self.num_slots)
        self.state[y : y + 2 * self.num_slots, 0:x] = C_LEGEND / 30

        y, x = (int((self.env.num_rows + 0.5) * self.num_slots), 2 * self.num_slots)
        self._draw_transaction(x, y, self.env.staging_in_area, True)

        y, x = (int((self.env.num_rows + 0.5) * self.num_slots), 4 * self.num_slots)
        self._draw_transaction(x, y, self.env.staging_out_area, True)

        self.state_history.append(self.state)

    def save_video(self):
        """
        Generate video out of self.state_history and save it. This variable
        needs to be updated during the simulation.
        """
        import matplotlib.pyplot as plt
        from moviepy.editor import VideoClip
        from moviepy.video.io.bindings import mplfig_to_npimage

        history_of_states = self.state_history
        # duration_in_seconds = len(history_of_states) / 4
        duration_in_seconds = len(history_of_states)
        fig, ax = plt.subplots()
        frames_per_second = len(history_of_states) / duration_in_seconds

        def make_frame(t):
            ax.clear()
            ax.grid(False)

            ax.imshow(history_of_states[int(t * frames_per_second)], cmap="gist_ncar")
            ax.tick_params(
                axis="both",
                which="both",
                bottom=False,
                top=False,
                left=False,
                right=False,
                labelleft=False,
                labelbottom=False,
            )
            return mplfig_to_npimage(fig)

        animation = VideoClip(make_frame, duration=duration_in_seconds)
        animation.write_videofile(self.env.video_filename, fps=frames_per_second)
