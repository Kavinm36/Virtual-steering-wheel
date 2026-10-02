# Virtual Steering Wheel Pro 🏎️💨

AI-powered Virtual Steering Wheel using MediaPipe Hand Tracking, OpenCV, and PyDirectInput. Designed for smooth, responsive driving in both **online web browser games** and **real PC racing games** like *Need for Speed: Most Wanted*.

---

## 🎮 Supported Games

### 1. Real PC Games (DirectX / DirectInput)
- **Need for Speed: Most Wanted (2005 & 2012)**
- **Need for Speed: Underground 1 & 2 / Carbon / Heat**
- **Grand Theft Auto V / San Andreas**
- **Forza Horizon / Assetto Corsa** (Keyboard mode)

### 2. Online Browser Games (HTML5 / WebGL)
- **Slow Roads** (slowroads.io)
- **Drift Hunters**
- **Smash Karts**
- Any racing or driving game in Chrome, Edge, Firefox, or Brave.

---

## 🛠️ Controls & Gestures

| Action | Physical Gesture | Default Mapping | WASD Preset | Arrow Keys Preset (NFS MW) |
|---|---|---|---|---|
| **Steer Left** | Tilt hands Counter-Clockwise (Left hand down, Right hand up) | Both Hands | `A` | `Left Arrow` |
| **Steer Right** | Tilt hands Clockwise (Left hand up, Right hand down) | Both Hands | `D` | `Right Arrow` |
| **Accelerate (Gas)** | **Thumb DOWN** (point down towards wrist) | **Left Hand** | `W` | `Up Arrow` |
| **Brake / Reverse** | **Thumb DOWN** (point down towards wrist) | **Right Hand** | `S` | `Down Arrow` |
| **Nitro (NOS)** | **Pinky EXTENDED** (open pinky finger) | **Right Hand** | `Space` | `Left Shift` |

> [!NOTE]
> If you prefer **Right Hand = Gas** and **Left Hand = Brake** (like real car foot pedals), simply press **`H`** inside the camera window to swap them!

---

## ⌨️ In-Game Hotkeys (Press inside the Steering Wheel window)

- **`M`** — **Switch Profile**: Toggle between `WASD (Browser)` and `ARROWS (NFS Most Wanted)`.
- **`H`** — **Swap Gas/Brake Hands**: Toggle between `Left=Gas, Right=Brake` and `Right=Gas, Left=Brake`.
- **`T`** — **Toggle Steering Mode**:
  - `PWM_SMOOTH`: Proportional pulsing for realistic cornering without spin-outs.
  - `DIRECT_HOLD`: Instant key lock for classic arcade games.
- **`N`** — **Toggle Nitro Gesture**: Enable/disable pinky nitro trigger.
- **`ESC`** — **Exit**: Safely release all keys and shut down.

---

## 🚀 How to Run

Simply double-click **`run.bat`**!

`run.bat` automatically detects the local embedded Python environment (`python312-embed`) with all required packages pre-installed.

---

## 🏆 Tips for Maximum Accuracy & NFS Most Wanted Setup

1. **Run as Administrator for NFS Most Wanted**:
   - If your copy of *Need for Speed: Most Wanted* is set to "Run as Administrator", Windows blocks inputs from non-admin apps (UIPI).
   - If the game does not respond to steering, right-click `run.bat` and select **"Run as Administrator"**.

2. **Webcam Positioning & Lighting**:
   - Place your webcam directly in front of you at chest or chin height.
   - Keep a uniform light source in front of you (avoid having a bright window behind you).
   - Hold your hands out at about steering-wheel width (25–35 cm apart).

3. **In-Game Settings for NFS Most Wanted**:
   - **Steering Sensitivity**: Set steering sensitivity in NFS MW to 70–80% for instant response.
   - **Speed Steering**: In NFS MW options, keep speed steering damping ON to prevent high-speed twitching.

4. **Proportional PWM Steering vs Direct Hold**:
   - On highways and slight bends, tilt gently ($8^\circ$ to $20^\circ$) for smooth line tracking.
   - For 90-degree street corners or hairpin drifts, tilt $>30^\circ$ to lock the wheels into a full turn!
