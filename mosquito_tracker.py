# Tracking class and logic

import torch

class MosquitoTracker:
    def __init__(self, min_hits=3, max_age=5, threshold_radius=30, frame_size=(1920, 1080, 3), device='mps'):
        self.tracks = {}
        self.next_id = 0
        self.min_hits = min_hits
        self.max_age = max_age
        self.threshold_radius = threshold_radius
        self.id_map = {}
        self.display_counter = 1
        self.frame_size = frame_size
        self.device = device
    
    def remove_duplicate_bboxes(self, centers, conf, duplicate_threshold=20):
        if centers.numel() == 0:
            return torch.empty((0, 2), device=self.device), []
        # (N, N) матрица расстояний между центрами
        dists = torch.cdist(centers, centers)

        # Матрица "дубликатов" — True, если i и j ближе duplicate_threshold
        close_mask = (dists < duplicate_threshold) & ~torch.eye(len(centers), dtype=torch.bool, device=self.device)

        conf_diff = conf[:, None] - conf[None, :]  # (N, N)

        # Для каждой пары (i, j): если j ближе и conf[j] > conf[i] → i удалить
        remove_mask = close_mask & (conf_diff < 0)

        # Если хоть один j "победил" i → удаляем i
        remove_any = remove_mask.any(dim=1)
        keep = ~remove_any
        return centers[keep], conf[keep]

    """Обновляет существующие треки: сопоставляет с детекциями или предсказывает позицию."""
    def _update_existing_tracks(self, det_centers, updated_ids):
        det_centers = det_centers.clone()
        for track_id, track in list(self.tracks.items()):
            track["age"] += 1
            # Если есть детекции, пытаемся сопоставить
            if len(det_centers) > 0:
                dists = torch.norm(det_centers - track["center"], dim=1)
                min_idx = torch.argmin(dists)

                if dists[min_idx] < self.threshold_radius:
                    i = min_idx.item()
                    velocity = det_centers[i] - track["center"]
                    track["velocity"] = velocity
                    track["center"] = det_centers[i].clone()
                    track["history"].append(track["center"])
                    track["hits"] += 1
                    track["age"] = 0
                    updated_ids.add(i)
                    det_centers[i] = torch.tensor([float('inf'), float('inf')], device=self.device)
                    continue  # Переходим к следующему треку

            # Если нет детекций или не сопоставлено, предсказываем позицию
            if torch.norm(track["velocity"]) > 0:
                predicted_center = track["center"] + track["velocity"]
                track["center"] = predicted_center
                track["history"].append(track["center"])

            self.delete_bad_tracks(track_id, track)

    def delete_bad_tracks(self, track_id, track):
        age_check = track["age"] > self.max_age
        position_check = not (0 <= track["center"][0] < self.frame_size[0] and 0 <= track["center"][1] < self.frame_size[1])

        if age_check or position_check:
            del self.tracks[track_id]
            if track_id in self.id_map:
                del self.id_map[track_id]




    def _add_new_tracks(self, centers, updated_ids):
        """Добавляет новые треки для несопоставленных детекций."""
        for i, center in enumerate(centers):
            if i not in updated_ids:
                self.tracks[self.next_id] = {
                    "center": center,
                    "velocity": torch.tensor([0.0, 0.0]),
                    "hits": 1,
                    "age": 0,
                    "history": [center]
                }
                self.next_id += 1

    def _get_confirmed_tracks(self):
        """Возвращает подтверждённые треки с display_id."""
        confirmed = []
        for track_id, track in self.tracks.items():
            if track["hits"] >= self.min_hits:
                if track_id not in self.id_map:
                    self.id_map[track_id] = self.display_counter
                    self.display_counter += 1
                display_id = self.id_map[track_id]
                confirmed.append((display_id, track["center"]))
        return confirmed

    def update(self, centers, conf):
        centers, conf = self.remove_duplicate_bboxes(centers, conf)
        updated_ids = set()
        self._update_existing_tracks(centers, updated_ids)
        self._add_new_tracks(centers, updated_ids)
        return self._get_confirmed_tracks()
    