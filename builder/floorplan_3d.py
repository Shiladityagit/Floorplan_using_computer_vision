import numpy as np
import trimesh
from trimesh.transformations import rotation_matrix

def create_wall(extents, position, color):
    w = trimesh.creation.box(extents=extents)
    w.visual = trimesh.visual.ColorVisuals(w, vertex_colors=color)
    w.apply_translation(position)
    return w

def create_door(extents, position, origin, color, hinges, rotate_clockwise):
    d = trimesh.creation.box(extents=extents)
    d.visual = trimesh.visual.ColorVisuals(d, vertex_colors=color)
    d.apply_translation(position)
    hinges.append((d, origin, rotate_clockwise))
    return d

def create_3d_model(results, class_names):
    sf = 0.1
    rh = 3.0
    wall_t = 0.2
    dh = 2.5
    dw = 1.0
    dd = 0.05
    da = np.pi / 2
    wc = [150, 150, 150, 255]
    dc = [139, 69, 19, 255]

    walls, doors, hinges = [], [], []
    wall_pos = []
    room, door = [], []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls = class_names[int(box.cls[0])]
            if "door" in cls.lower():
                door.append([x1, y1, x2, y2])
            else:
                room.append({"class": cls, "bbox": [x1, y1, x2, y2]})

    for det in room:
        x1, y1, x2, y2 = np.array(det["bbox"]) * sf
        walls += [
            create_wall((x2 - x1, rh, wall_t), [(x1 + x2) / 2, rh / 2, y1], wc),
            create_wall((x2 - x1, rh, wall_t), [(x1 + x2) / 2, rh / 2, y2], wc),
            create_wall((wall_t, rh, y2 - y1), [x1, rh / 2, (y1 + y2) / 2], wc),
            create_wall((wall_t, rh, y2 - y1), [x2, rh / 2, (y1 + y2) / 2], wc)
        ]
        wall_pos += [
            ((x1 + x2) / 2, y1, "H"),
            ((x1 + x2) / 2, y2, "H"),
            (x1, (y1 + y2) / 2, "V"),
            (x2, (y1 + y2) / 2, "V")
        ]

    for dx1, dy1, dx2, dy2 in door:
        dx1, dy1, dx2, dy2 = np.array([dx1, dy1, dx2, dy2]) * sf
        is_h = abs(dx2 - dx1) > abs(dy2 - dy1)
        min_d, cw = float("inf"), None

        for wx, wy, t in wall_pos:
            if (is_h and t == "H") or (not is_h and t == "V"):
                d = abs((wy if is_h else wx) - (dy1 if is_h else dx1))
                if d < min_d:
                    min_d = d
                    cw = (wx, wy, t)

        if cw:
            wx, wy, t = cw
            pos = [dx1, dh / 2, wy] if t == "H" else [wx, dh / 2, dy1]
            ext = (dw, dh, dd) if t == "H" else (dd, dh, dw)
            doors.append(create_door(ext, pos, pos, dc, hinges, True))

    floor = trimesh.creation.box(extents=(100, 0.1, 100))
    floor.visual = trimesh.visual.ColorVisuals(floor, vertex_colors=[200, 200, 200, 255])
    floor.apply_translation([50, -0.05, 50])

    for d, h, c in hinges:
        a = [0, 1, 0]
        r = rotation_matrix(-da if c else da, a, h)
        d.apply_transform(r)

    return trimesh.Scene([floor] + walls + doors)
