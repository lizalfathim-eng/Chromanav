# ChromaNav

ChromaNav is a desktop prototype that uses a webcam, a YOLO object-detection
model, and spoken announcements to provide additional road awareness. It is
intended as an assistive aid, including for colour-blind users: when the model
detects a traffic light, ChromaNav attempts to identify its colour and speak
the result instead of relying on colour alone.

ChromaNav does not interpret every road sign or traffic situation. It reports
only the limited classes described below, and its output is not a substitute
for looking at the road or making independent safety decisions.

## Current Detection Scope

The bundled `yolov8n.pt` checkpoint is a COCO-trained model with 80 classes.
Although it can produce detections for other COCO objects, the app maps and
processes only these two exact model classes:

| Model class | App label and behavior |
| --- | --- |
| `traffic light` | The traffic-light crop is analyzed using HSV colour ranges and a heuristic based on the usual vertical signal order. ChromaNav can display and announce `Red Light`, `Yellow Light`, or `Green Light`. If it cannot determine a colour, it uses `Traffic Light`. |
| `stop sign` | Displayed and announced as `Stop`. This is the only road-sign class currently handled by the app. |

The light colours are estimated from pixels in the detected box; they are not
separate classes in the bundled YOLO model. The current model and mapping do
not support Indian road-sign coverage beyond its `stop sign` class. Other
model classes and unrecognized labels are ignored.

## Detection Workflow

1. **Camera input:** Starting detection opens camera device `0` (the default
   webcam) through OpenCV. The app does not currently provide a video-file
   input option.
2. **Object detection:** The loaded Ultralytics YOLO model runs on each camera
   frame at a 640-pixel inference size. The loop is limited to at most 8
   inferences per second.
3. **Class mapping:** The model's class name is normalized and looked up in
   `SUPPORTED_DETECTION_LABELS`. Only `traffic light` and `stop sign` are
   mapped; unmatched classes are skipped.
4. **Confidence filtering:** YOLO inference uses a 0.50 confidence threshold,
   and each supported app label is also checked against its 0.50 threshold in
   `CONFIDENCE_THRESHOLDS`. This reduces low-confidence detections; it cannot
   eliminate false positives.
5. **Traffic-light colour estimation:** For a mapped traffic-light detection,
   OpenCV applies HSV colour masks to the cropped box and checks the expected
   vertical signal regions. If no colour passes its pixel-count checks, the
   label remains `Traffic Light`.
6. **Display and speech:** Accepted detections are outlined and labelled in
   the OpenCV video window. The detection thread sends announcement text to a
   priority queue. A separate `pyttsx3` worker speaks queued messages
   sequentially. Repeated announcements are rate-limited with cooldowns:
   three seconds for traffic-light labels and five seconds for other supported
   labels.

## Installation and Usage

### Requirements

- Python
- `opencv-python`
- `numpy`
- `pyttsx3`
- `ultralytics`
- A working webcam and an audio output device; a microphone is not required

Tkinter is used for the desktop controls and is included with many standard
Python installations. On Windows, speech uses the SAPI5 voice engine; on other
platforms, `pyttsx3` uses the platform's available speech backend.

Install the Python packages from PowerShell:

```powershell
python -m pip install opencv-python numpy pyttsx3 ultralytics
```

### Model weights

By default, the app loads `yolov8n.pt` from the same directory as
`chromanav.py`. The weight file is not tracked in this repository, so make
sure it is present locally before starting detection. The application does
not download or train custom weights as part of its setup.

To select a different local Ultralytics-compatible YOLO checkpoint, set
`CHROMANAV_MODEL` before launching the app:

```powershell
$env:CHROMANAV_MODEL = "C:\path\to\your-model.pt"
python chromanav.py
```

Restart the app after changing the environment variable. The loaded model
must expose class names that the app maps; changing weights alone does not
automatically enable their classes.

### Run the application

From the project directory:

```powershell
python chromanav.py
```

In the ChromaNav window:

- Select **Test Voice** to check that speech output is working.
- Select **Start Detection** to load the model and start the webcam.
- Select **Stop Detection** to stop detection, or press `q` while the video
  window is focused.
- Check the status labels for camera/model errors and speech-engine status.

Allow the operating system to access the camera and audio output. Verify the
voice and camera in a safe, stationary setting before using the application.

## Extending Detection

### When mapping is enough

Adding a label mapping is appropriate only when the loaded model already
detects the desired class and exposes its class name in `model.names`. For
example, if a compatible checkpoint was trained to detect `no entry`, adding
a mapping for that exact class name can let the app display and announce it.
A mapping changes how an existing detection is presented; it does not teach a
model a new visual category.

To enable an already-trained class:

1. Inspect the checkpoint's class names and confirm the class is actually
   present.
2. Add its normalized model class name and desired display/speech label to
   `SUPPORTED_DETECTION_LABELS` in `chromanav.py`.
3. Add a confidence threshold for the resulting app label to
   `CONFIDENCE_THRESHOLDS`.
4. Test with representative images or video and tune the threshold based on
   observed false positives and missed detections.

The app uses the mapped label as the spoken text, so a supported label is
automatically included in speech announcements. Repeated detections use the
existing five-second general announcement cooldown. If the class needs a
different spoken phrase, update the label or the speech-message construction
as well.

### When training or fine-tuning is needed

If a model does not detect a sign class, a mapping or a lower confidence
threshold is not enough. Collect or obtain a suitable labelled dataset,
including representative road conditions and examples of the target sign,
then train or fine-tune a compatible object-detection model. Validate the
result on separate data before using it. Add mappings only for the class names
present in that checkpoint, and document the model and its actual class
coverage. ChromaNav itself currently has no dataset preparation or model
training workflow.

### Using alternative pretrained YOLO weights

Set `CHROMANAV_MODEL` to the checkpoint path as shown in [Model
weights](#model-weights). The current app keeps class handling modular through
`SUPPORTED_DETECTION_LABELS` and per-label values in
`CONFIDENCE_THRESHOLDS`. Alternative weights can be loaded, but only classes
whose names are mapped by the app are displayed or announced. Checkpoint
compatibility, class names, and output quality must be verified for each
model; a checkpoint trained for another domain is not automatically suitable
for traffic signs.

## Limitations and Safety

- Detection is limited to the classes supported by the loaded model **and**
  mapped by the app. The bundled model's only handled road-sign class is
  `stop sign`; ChromaNav does not currently provide comprehensive Indian road
  sign detection.
- Results depend on model training data, camera quality, lighting, sign or
  signal size and distance, occlusion, viewing angle, and other environmental
  conditions.
- False positives and missed detections are possible. A confidence threshold
  reduces some low-confidence detections but does not guarantee correctness.
- Traffic-light colour estimation is a simple HSV and vertical-position
  heuristic. It can be uncertain in poor lighting, with small or occluded
  signals, or when a signal is not vertically arranged.
- Speech is handled by a single worker, so announcements do not speak over one
  another. They may be delayed while an earlier phrase is playing; detections
  can be queued, and a voice test is prioritized over queued detection
  messages but does not interrupt a phrase already being spoken.
- ChromaNav is a prototype, not a certified road-safety system. Do not rely on
  it as the sole source of road information. It must not replace visual
  awareness, traffic rules, or independent road-safety decisions.

## Future Improvements

These are ideas for future development, not features currently provided:

- Support additional Indian road signs through appropriately trained and
  validated models.
- Improve detection accuracy and robustness in varied lighting, distance, and
  camera conditions.
- Add multilingual voice announcements.
- Improve real-time performance and announcement prioritization.
- Add accessibility options such as configurable speech, clearer status
  feedback, and alternative visual/audio cues.
