"""Live overlay: the current frame plus every question's latest answer, its tier and its age.
OpenCV windows must run on the main thread on macOS; press q to quit, m to toggle mirroring.

Mirroring flips only the preview. The model always sees the camera's own view, so directional answers
("moving to the right") refer to the unmirrored image."""
import cv2
import numpy as np

COLORS = {"yes": (60, 200, 60), "no": (70, 70, 230), "uncertain": (160, 160, 160)}


def _age_color(age_ms):
    return (220, 220, 220) if age_ms < 500 else (0, 190, 255) if age_ms < 1500 else (60, 60, 255)


def render(runtime, bus, width=960, mirror=False):
    latest = runtime.latest_frame
    if latest is None:
        return None
    frame, t_now = latest
    h, w = frame.shape[:2]
    canvas = cv2.cvtColor(cv2.resize(frame, (width, int(h * width / w))), cv2.COLOR_RGB2BGR)
    if mirror:
        canvas = cv2.flip(canvas, 1)  # preview only; the answer panel is drawn unflipped
    decisions = sorted(bus.latest().values(), key=lambda d: (d.tier, d.question_id))
    panel = np.zeros((canvas.shape[0], 430, 3), dtype=np.uint8)

    ticks = runtime.stats["tick_ms"]
    t2 = runtime.stats["tier2_times"]
    rate = (len(t2) - 1) / (t2[-1] - t2[0]) if len(t2) > 1 and t2[-1] > t2[0] else 0.0
    header = f"tick p50 {np.median(ticks):.0f} ms | tier2 {rate:.1f} Hz | drops {runtime.stats['dropped_groups']}" \
        if ticks else "warming up..."
    cv2.putText(panel, header, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    y = 52
    for d in decisions:
        if d.tier == 0 and d.answer == "no":
            continue  # show camera-health flags only when raised
        age = (t_now - d.t_end) * 1e3 if runtime.realtime else d.age_ms
        color = COLORS.get(d.answer, (0, 215, 255))
        cv2.putText(panel, f"T{d.tier} {d.question_id[:22]:<22}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (230, 230, 230), 1, cv2.LINE_AA)
        cv2.putText(panel, f"{d.answer[:14]} {d.p:.2f}", (215, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        cv2.putText(panel, f"{age:4.0f}ms", (360, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, _age_color(age), 1, cv2.LINE_AA)
        y += 22
        if y > panel.shape[0] - 10:
            break
    return np.hstack([canvas, panel])


def show(runtime, bus, stop, threads, title="vertix", mirror=False):
    while not stop.is_set() and any(th.is_alive() for th in threads):
        img = render(runtime, bus, mirror=mirror)
        if img is not None:
            cv2.imshow(title, img)
        key = cv2.waitKey(15) & 0xFF
        if key == ord("q"):
            stop.set()
        elif key == ord("m"):
            mirror = not mirror
    cv2.destroyAllWindows()
