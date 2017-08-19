using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class RandomScript : MonoBehaviour
{
    public GameObject capcam;
    public GameObject maincam;

    private Vector3 camAngles;
    //private Material skybox;
    //private float exposure;

    private Color RandomColorDelta(float scale)
    {
        return new Color(Random.Range(-scale, scale), Random.Range(-scale, scale), Random.Range(-scale, scale), 0.0f);
    }

    public void Randomize(int seed, float scale = 1.0f)
    {
        Random.InitState(seed);

        // Change properties of lighting and sky.
        //RenderSettings.skybox.SetFloat("_Exposure", exposure + Random.Range(-0.5f, 2.0f) * scale);

        capcam.transform.localEulerAngles = camAngles + new Vector3(Random.Range(-4.0f, 4.0f), Random.Range(-4.0f, 4.0f), Random.Range(-4.0f, 4.0f)) * scale;
        maincam.transform.localEulerAngles = camAngles + new Vector3(Random.Range(-4.0f, 4.0f), Random.Range(-4.0f, 4.0f), Random.Range(-4.0f, 4.0f)) * scale;
    }

    void SaveStartState()
    {
        //exposure = RenderSettings.skybox.GetFloat("_Exposure");

        if (capcam == null) capcam = GameObject.Find("/car_root/CameraCapture");
        if (maincam == null) maincam = GameObject.Find("/car_root/CameraMain");
        camAngles = capcam.transform.localEulerAngles;
    }

    // Use this for initialization
    void Start()
    {
        SaveStartState();
        Randomize(0, 0.0f);
    }

    // Update is called once per frame
    void Update()
    {
        //		if ((Time.frameCount % 15) == 0)
        //			Randomize(Random.Range(0, 32000));
    }
}