# ChromaNav

ChromaNav uses a YOLO object-detection model and a webcam to label supported
traffic lights and road signs in a live video window. It announces detections
with text-to-speech.

## Setup

Install Python dependencies:

```powershell
python -m pip install opencv-python numpy pyttsx3 ultralytics
```

Run the app:

```powershell
python chromanav.py
```

On the first run, Ultralytics may download the default `yolov8n.pt` model.
Allow camera and audio access. Press **Start Detection** in the app; press
**Stop Detection** to stop, or press `q` in the video window. Use **Test Voice**
to check audio before relying on spoken announcements. The app shows voice
engine errors separately from camera/model errors. During detection, the
status shows when a label is sent to speech; the voice status shows the label
currently being spoken and the last label completed. A voice test requested
during detection is prioritized ahead of queued detections and plays after
the current spoken phrase.

## Detection model support

The default `yolov8n.pt` model is trained on COCO. It can identify traffic
lights and stop signs, but it does not recognize the other road-sign types
listed in this app. Those signs require a YOLO model trained on a suitable
traffic-sign dataset. Class names such as `Speed Limit 30`, `No Parking`,
`Pedestrian Crossing`, and `Road Work` are mapped to spoken/display labels.
Other model classes are ignored rather than incorrectly reported as signs.
The app runs inference at 960 pixels with a 0.25 confidence threshold to
improve detection of small or distant objects. This may run slower on older
computers. These settings cannot make a model detect classes it was not
trained on; a traffic-sign-specific model is required for those signs.

To use custom model weights, set `CHROMANAV_MODEL` to the weights file before
starting the app:

```powershell
$env:CHROMANAV_MODEL = "C:\path\to\traffic-signs.pt"
python chromanav.py
```

Detection quality depends on the model, camera, lighting, and training data.
Traffic-light color uses both HSV color and the usual vertical signal order
(red at top, yellow in the middle, green at bottom). It can still be uncertain
in poor lighting or when the light is small, occluded, or the signal is not
vertically arranged. Treat the output as an assistive aid, not as a substitute
for observing the road and traffic rules.
