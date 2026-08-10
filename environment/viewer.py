from __future__ import annotations

import pygame

from environment.gridworld import GridWorld

ACTION_NAMES = {
    0: "UP",
    1: "DOWN",
    2: "LEFT",
    3: "RIGHT",
}


class PygameViewer:
    def __init__(self, grid_size: int, cell_size: int = 64):
        pygame.init()

        self.grid_size = grid_size
        self.cell_size = cell_size
        panel_size = 260
        width = grid_size * cell_size + panel_size
        height = grid_size * cell_size
        self.width = width
        self.font = pygame.font.SysFont(None, 28)
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("GridWorld MLP Evaluation")

        self.clock = pygame.time.Clock()

    def process_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        return True

    def draw_text(self, text: str, x: int, y: int) -> None:
        surface = self.font.render(text, True, "black")
        self.screen.blit(surface, (x, y))

    def render(
        self,
        env: GridWorld,
        step: int,
        model_action: int,
        oracle_action: int,
        executed_action: int,
        interventions: int,
        status: str = "RUNNING",
    ) -> None:
        self.screen.fill("white")

        grid = env.observation_grid()

        for row in range(env.grid_size):
            for col in range(env.grid_size):
                value = grid[row, col]

                rect = pygame.Rect(
                    col * self.cell_size,
                    row * self.cell_size,
                    self.cell_size,
                    self.cell_size,
                )

                if value == -1:
                    color = "black"
                elif value == 1:
                    color = "blue"
                elif value == 2:
                    color = "green"
                else:
                    color = "white"

                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, "gray", rect, 1)

        panel_x = self.grid_size * self.cell_size
        panel_rect = pygame.Rect(panel_x, 0, self.width, self.screen.get_height())
        pygame.draw.rect(self.screen, "lightgray", panel_rect)
        x = panel_x + 20
        y = 30
        gap = 35

        self.draw_text(f"Step: {step}", x, y)
        self.draw_text(f"Model:{ACTION_NAMES[model_action]}", x, y + gap)
        self.draw_text(f"Oracle:{ACTION_NAMES[oracle_action]}", x, y + gap * 2)
        self.draw_text(f"Executed:{ACTION_NAMES[executed_action]}", x, y + gap * 3)
        self.draw_text(f"Interventions: {interventions}", x, y + gap * 4)
        self.draw_text(f"Status: {status}", x, y + gap * 6)
        self.draw_text(f"Model:{ACTION_NAMES[model_action]}", x, y + gap)
        pygame.display.flip()

    def render_grid(self, env: GridWorld):
        grid = env.observation_grid()
        for row in range(env.grid_size):
            for col in range(env.grid_size):
                value = grid[row, col]

                rect = pygame.Rect(
                    col * self.cell_size,
                    row * self.cell_size,
                    self.cell_size,
                    self.cell_size,
                )

                if value == -1:
                    color = "black"
                elif value == 1:
                    color = "blue"
                elif value == 2:
                    color = "green"
                else:
                    color = "white"

                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, "gray", rect, 1)
        pygame.display.flip()

    def tick(self, fps: int) -> None:
        self.clock.tick(fps)

    def close(self):
        pygame.quit()
