"""Step 1: Record labeled hand-landmark samples.

Example:
  python collect_data.py --letter A --samples 300 --person p1 --session s1 --lighting normal --distance medium

Press SPACE to start recording the letter, Q to quit.
Repeat for every letter, for several people, sessions, lighting and distances.
"""
import argparse
import csv
import os

import cv2
import mediapipe as mp

from features import FEATURE_COLUMNS, get_hand_features

META = ["label", "person", "session", "lighting", "distance"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--letter", required=True)
    ap.add_argument("--samples", type=int, default=300)
    ap.add_argument("--person", default="p1")
    ap.add_argument("--session", default="s1")
    ap.add_argument("--lighting", default="normal", choices=["bright", "normal", "dim"])
    ap.add_argument("--distance", default="medium", choices=["near", "medium", "far"])
    ap.add_argument("--out", default="data/landmarks.csv")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--skip", type=int, default=2, help="save every Nth frame for more variety")
    args = ap.parse_args()

    letter = args.letter.upper()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    new_file = not os.path.exists(args.out)

    mp_hands, mp_draw = mp.solutions.hands, mp.solutions.drawing_utils
    cap = cv2.VideoCapture(args.camera)
    recording, saved, frame_no = False, 0, 0

    with open(args.out, "a", newline="") as f, mp_hands.Hands(
        max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.5
    ) as hands:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(META + FEATURE_COLUMNS)

        while cap.isOpened() and saved < args.samples:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            feats, lms = get_hand_features(results)

            if lms is not None:
                mp_draw.draw_landmarks(frame, lms, mp_hands.HAND_CONNECTIONS)
                frame_no += 1
                if recording and feats is not None and frame_no % args.skip == 0:
                    writer.writerow(
                        [letter, args.person, args.session, args.lighting, args.distance] + feats.tolist()
                    )
                    saved += 1

            status = f"Letter {letter} | saved {saved}/{args.samples}"
            hint = "RECORDING - vary angle slightly" if recording else "Press SPACE to start, Q to quit"
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, hint, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            cv2.imshow("Collect data", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                recording = True

    cap.release()
    cv2.destroyAllWindows()
    print(f"Saved {saved} samples for {letter} to {args.out}")


if __name__ == "__main__":
    main()