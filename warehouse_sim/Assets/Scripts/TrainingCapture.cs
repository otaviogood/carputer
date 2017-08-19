using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

public class TrainingCapture : MonoBehaviour
{
    public Camera target;
    public CarController car;
    public int width = 128;
    public int height = 128;
    public float rate = 30.0f;
    public ExternalPipe pipe;
    public RandomScript randomizer;

    private RenderTexture renderTarget;
    private Texture2D texture;
    private bool capturing;
    private bool sendDataOverPipe = false;
    private float elapsed;
    private string baseDir;
    private int captureCount;

    void Start()
    {
        renderTarget = new RenderTexture(width, height, 24);
        renderTarget.antiAliasing = 8;
        renderTarget.Create();
        target.targetTexture = renderTarget;
        texture = new Texture2D(width, height, TextureFormat.RGB24, false);
    }

    private void Update()
    {
        if (Input.GetKeyUp(KeyCode.C))
        {
            capturing = !capturing;
            if (capturing)
            {
                string docDir = System.Environment.GetFolderPath(System.Environment.SpecialFolder.MyDocuments);
                baseDir = Path.Combine(Path.Combine(docDir, "carputer_training"), DateTime.Now.ToString("yy-MM-dd-HH-mm-ss"));
                System.IO.Directory.CreateDirectory(baseDir);
                elapsed = 0;
                captureCount = 0;
            }
        }
    }

    void LateUpdate()
    {
        if (capturing || sendDataOverPipe)
        {
            //var randomizer = GameObject.Find("/Randomizer").GetComponent<RandomizerScript>();
            randomizer.Randomize(new System.Random().Next(), 1.0f);

            float interval = (1.0f / rate);
            elapsed += Time.deltaTime;

            if (elapsed > interval)
            {
                RenderTexture.active = renderTarget;
                target.Render();
                texture.ReadPixels(new Rect(0, 0, width, height), 0, 0);
                RenderTexture.active = null;

                //double lat = 0.0;
                //double lon = 0.0;
                //GPSAnchorScript.GetLatLong(car.gameObject, out lat, out lon);

                // Write the captured frame to a file
                if (capturing)
                {
                    byte[] bytes = texture.EncodeToPNG();// .EncodeToJPG(80);
                    System.IO.File.WriteAllBytes(Path.Combine(baseDir, String.Format("frame_{0}_thr_{1}_ste_{2}_mil_{3}_odo_{4}_.png", captureCount.ToString().PadLeft(5, '0'), (int)car.throttle, (int)car.wheelAngle, (int)(Time.time * 1000), (int)car.odometer)), bytes);
                    captureCount++;
                }

                // Send the metadata and image to tensorflow
                if (sendDataOverPipe)
                {
                    float[] floats = new float[] {
                        width,
                        height,
                        car.wheelAngle,
                        car.throttle,
                        (float)0,//lat,
                        (float)0,//lon,
                    };
                    Color32[] pixels = texture.GetPixels32();
                    int pixelCount = width * height;

                    // Build and send the message
                    byte[] message = new byte[floats.Length * 4 + pixelCount * 3];
                    Buffer.BlockCopy(floats, 0, message, 0, floats.Length * 4);
                    for (int i = 0, j = floats.Length * 4; i < pixelCount; i++, j += 3)
                    {
                        Color32 pixel = pixels[i];
                        message[j] = pixel.r;
                        message[j + 1] = pixel.g;
                        message[j + 2] = pixel.b;
                    }
                    pipe.Send(message);
                }

                elapsed = interval - elapsed;
            }

            randomizer.Randomize(0, 0.0f);
        }
    }
}