# Science Camera Image APK

This is an Android guest payload application that takes full-resolution
pictures with the science camera.

The pictures are published on the `/hw/cam_sci/compressed` topic via
ROS at reduced resolution (default: 640x480). The image dimensions for
the published image is published on the `/hw/cam_sci_info` topic.

The full-resolution images (at 5344x4008 pixels) are saved locally on
GC in directory:

  `/sdcard/data/gov.agency.fac.team.cabinflyer.imaging_camera_image/delayed`

and can be fetched later.

This app has no GUI. It should be invoked via the guest payload
manager.

## Setting up the environment

Set the variables that point to your local copy of the cabinflyer main
repository and to the cabinflyer android repository, for example, as:

    export SOURCE_PATH=$HOME/cabinflyer
    export ANDROID_PATH=$SOURCE_PATH/submodules/android

## Building the app

It is expected that you will have read `$ANDROID_PATH/build_essential_apks.md`
and `$ANDROID_PATH/running_gs_app.md` for background information.

Run on your development machine:

    cd $ANDROID_PATH/guest_payload_projects/imaging_camera_image
    ANDROID_HOME=$HOME/Android/Sdk ./gradlew assembleDebug

After building the apk, you will probably want to copy it to the robot. Please
copy it to the CC processor. For example:

    rsync -avzP app/build/outputs/apk/app-debug.apk dogwood-cc:imaging_camera_image.apk

## Ensure the robot has the correct time

Connect to CC and run the `date` command. From there, connect to GC,
via the:

    adb shell

command and run the `date` command as well. If these do not agree, please sync
the processors. To do this, run the following command on groundhost or computer:

    $(date '+%s.%N')

Then pass the result to the set time script. This must be run on the NC:

    /opt/cabinflyer/ops/set_time.bash %s.%N

## Setting up ROS communication

This app assumes sets the ROS master URI as

    http://cc:11311

If that is not the case, the code must be edited to set the correct
value and rebuilt. You may need to also set the environmental variable
`ROS_MASTER_URI` to the same value in any shell that is used to do ROS
communication.

## Installing the imaging_camera_image APK

Connect to CC. Run:

    adb uninstall gov.agency.fac.team.cabinflyer.imaging_camera_image
    adb install -g imaging_camera_image.apk

(This will replace any older version of this app.)

## Running this APK using the Guest payload Manager

First, connect to CC. Then start the Cabinflyer fsw by running: 

    systemctl --user start fsw

Next, start the guest payload manager by running:

    /opt/cabinflyer/ops/gs_manager.sh start

Now use the console helper to start the science camera image apk, issue
commands to the apk, and stop it. First navigate to the helper:

    cd /opt/cabinflyer/ops/gds/

It is recommended to use the console helper GUI as it shows feedback for the
commands but all of the following commands can be used on the command line with
the batch console helper. To start the Console helper, run:

    python console_helper.py

To start the apk, run this line in the console helper:

    gs -start ImagingCameraImage

Once the science camera image apk is started, you can issue any of the following
commands using the console helper. For more information on what the commands do,
please see the next section.

    1) gs -cmd ImagingCameraImage '{"name": "takePicture"}'
    2) gs -cmd ImagingCameraImage '{"name": "setAutoExposure", "auto": true}'
    3) gs -cmd ImagingCameraImage '{"name": "setContinuousPictureTaking", "continuous": true}'
    4) gs -cmd ImagingCameraImage '{"name": "setFocusDistance", "distance": 0.39}'
    5) gs -cmd ImagingCameraImage '{"name": "setFocusMode", "mode": "manual"}'
    6) gs -cmd ImagingCameraImage '{"name": "setPublishImage", "publish": true}'
    7) gs -cmd ImagingCameraImage '{"name": "setPublishedImageSize", "width": 640, "height": 480}'
    8) gs -cmd ImagingCameraImage '{"name": "setPublishedImageType", "type": "color"}'
    9) gs -cmd ImagingCameraImage '{"name": "setSavePicturesToDisk", "save": true}'

To stop the apk, run this line in the console helper:

    gs -stop ImagingCameraImage

To close the console helper, please press F10. 

If you want to use the batch console helper on the command line, simply put one of
the above commands in quotes and give it to the batch helper. For example:

    python console_helper_batch.py cmd "gs -cmd ImagingCameraImage '{"name": "takePicture"}'"

To stop the guest payload manager (after stopping ImagingCameraImage), do:

    ./gs_manager.sh stop

If the guest payload manager is not behaving, one can use the option

    ./gs_manager.sh  hard_stop


## Description of the custom guest payload commands

1. Take Picture

    Take a single picture. If enabled, it will be published via ROS as a
    preview image of dimensions 640x480 pixels (this is customizable, see
    below), and saved to disk on the GC at full resolution.

    If continuous picture taking is on while issuing the command, the command
    will be rejected. The command may also be rejected if the apk is in the
    process of capturing an image from a previous take picture command. For
    reference, it takes 1.7 seconds for the apk to capture an image.

2. Set Auto Exposure

    Turns auto exposure on and off. By default, auto exposure is set to true.

3. Set continuous picture-taking

    Enable or disable capture images as fast as the camera will allow
    (typically one image per 1.7 seconds). The captured image will be published 
    via ROS and saved to disk if these options are enabled. By default this
    functionality is disabled.

4. Set focus distance

    Set the focus distance. The default is 0.39 (in units of 1/m)
    corresponding to the plane of best focus being approximately 1 m from
    the robot body. A value of 0 will result in the focus being set at
    infinity.

5. Set focus mode

    Set the focus mode as either 'manual' or 'auto'. By default, it is set to
    'manual', with the focus distance specified earlier.

6. Set publish image

    If this is set to true, the apk will publish captured images over ROS. The
    size and type are configurable with the next two commands. By default, the
    the images are published.

7. Set published image size

    Set the height and width of the images being published over ROS. The default
    size is 640 by 480 pixels.

8. Set published image type

    Set the published image type as 'color' or 'grayscale'. The default is set
    to 'color'.

9. Set save pictures to disk

    If this is set to true, the apk will write captured images to disk on the
    GC at full resolution. The directory where they are written was mentioned
    in an earlier section. By default, the images are saved to disk.

## Fetching the data

The imaging camera images can be retrieved from GC with the `adb pull`
command, followed by connecting to GC with `adb shell` and deleting
the images they are if no longer needed. The SURVEY project provides a
tool to add these images to a bag containing the other robot data
recorded at the same time.

## Logging with adb

To see logging info as this app is running one can do (in a separate
terminal on CC):

    adb logcat -b all -c   # wipe any existing logs
    adb logcat -s ImagingCameraImage
