# Copyright (c) 2026 The Cactus Authors


class AcceptanceRateTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0.0
        self.count = 0
        self.task = ""

    def update(self, new_ar):
        self.count += 1
        self.avg += (new_ar - self.avg) / self.count
        return self.avg


tracker = AcceptanceRateTracker()
