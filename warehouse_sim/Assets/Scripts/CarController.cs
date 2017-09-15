using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

[System.Serializable]
public class AxleInfo
{
    public WheelCollider leftWheel;
    public WheelCollider rightWheel;
    public bool motor;
    public bool steering;
}

public class CarController : MonoBehaviour
{
    //public List<AxleInfo> axleInfos;
    //public float maxMotorTorque;
    //public float maxSteeringAngle;

    //// finds the corresponding visual wheel
    //// correctly applies the transform
    //public void ApplyLocalPositionToVisuals(WheelCollider collider)
    //{
    //    if (collider.transform.childCount == 0)
    //    {
    //        return;
    //    }

    //    Transform visualWheel = collider.transform.GetChild(0);

    //    Vector3 position;
    //    Quaternion rotation;
    //    collider.GetWorldPose(out position, out rotation);

    //    visualWheel.transform.position = position;
    //    visualWheel.transform.rotation = rotation;
    //}

    //public void FixedUpdate()
    //{
    //    float motor = maxMotorTorque * Input.GetAxis("Vertical");
    //    float steering = maxSteeringAngle * Input.GetAxis("Horizontal");

    //    foreach (AxleInfo axleInfo in axleInfos)
    //    {
    //        if (axleInfo.steering)
    //        {
    //            axleInfo.leftWheel.steerAngle = steering;
    //            axleInfo.rightWheel.steerAngle = steering;
    //        }
    //        if (axleInfo.motor)
    //        {
    //            axleInfo.leftWheel.motorTorque = motor;
    //            axleInfo.rightWheel.motorTorque = motor;
    //        }
    //        ApplyLocalPositionToVisuals(axleInfo.leftWheel);
    //        ApplyLocalPositionToVisuals(axleInfo.rightWheel);
    //    }
    //}

    private float[] externalInputs;
    internal float steerMessage = 0.0f;

    private WheelCollider wheelColliderFrontLeft;
    private WheelCollider wheelColliderFrontRight;
    private WheelCollider wheelColliderRearLeft;
    private WheelCollider wheelColliderRearRight;

    private const float wheelAngleMax = 20.0f;
    private const float motorTorque = 600.0f;
    private const float brakeTorque = 1000.0f;
    internal float wheelAngle = 0.0f;
    internal float throttle = 0.0f;
    private float brake = 0.0f;
    internal float odometer = 0.0f;

    // Use this for initialization
    void Start() {
        //var wheelFrontLeft = GameObject.Find("/Car/WheelFrontLeft");
        //var wheelFrontRight = GameObject.Find("/Car/WheelFrontRight");
        //var wheelBackLeft = GameObject.Find("/Car/WheelBackLeft");
        //var wheelBackRight = GameObject.Find("/Car/WheelBackRight");
        //wheelColliderFrontLeft = wheelFrontLeft.GetComponent<WheelCollider>();
        //wheelColliderFrontRight = wheelFrontRight.GetComponent<WheelCollider>();
        //wheelColliderRearLeft = wheelBackLeft.GetComponent<WheelCollider>();
        //wheelColliderRearRight = wheelBackRight.GetComponent<WheelCollider>();
        //wheelColliderFrontLeft.ConfigureVehicleSubsteps(5f, 10, 10);
        //wheelColliderFrontRight.ConfigureVehicleSubsteps(5f, 10, 10);
        //wheelColliderRearLeft.ConfigureVehicleSubsteps(5f, 10, 10);
        //wheelColliderRearRight.ConfigureVehicleSubsteps(5f, 10, 10);
    }

    // Update is called once per frame
    void Update()
    {
        float horiz = Input.GetAxis("Horizontal");
        float vert = Input.GetAxis("Vertical");
        var rigid = this.GetComponent<Rigidbody>();
        var vel = rigid.velocity;
        vel = new Vector3(vel.z, vel.y, -vel.x);
        var dot = Vector3.Dot(vel, rigid.transform.forward);
        rigid.AddRelativeForce(new Vector3(1, 0, 0) * dot, ForceMode.VelocityChange);
        rigid.AddRelativeForce(new Vector3(0, 0, vert * 2.0f), ForceMode.Force);
        rigid.AddTorque(new Vector3(0, horiz * vel.magnitude * 1.5f, 0), ForceMode.Force);

        wheelAngle = (float)Math.Round(90.0f - horiz * 88.0f);
        throttle = (float)Math.Round(vert * 20.0f + 90.0f);
        //var camOb = GameObject.Find("/Camera");
        //var cam = camOb.GetComponent<Camera>();
        //cam.transform.LookAt(rigid.transform);
        odometer += rigid.velocity.magnitude * 14.0f * Time.deltaTime;  // Approximate
    }

    void UpdatePhysics()
    {
        //wheelColliderFrontLeft.steerAngle = wheelAngle;
        //wheelColliderFrontRight.steerAngle = wheelAngle;
        //wheelColliderFrontLeft.motorTorque = motorTorque * throttle;
        //wheelColliderFrontRight.motorTorque = motorTorque * throttle;

        //wheelColliderFrontLeft.brakeTorque = brakeTorque * brake;
        //wheelColliderFrontRight.brakeTorque = brakeTorque * brake;
        //wheelColliderRearLeft.brakeTorque = brakeTorque * brake;
        //wheelColliderRearRight.brakeTorque = brakeTorque * brake;
    }

    public void HandleExternalMessage(byte[] message)
    {
        externalInputs = new float[message.Length / 4];
        Buffer.BlockCopy(message, 0, externalInputs, 0, externalInputs.Length * 4);
        steerMessage = externalInputs[0];
        print("message");
        foreach (var v in externalInputs)
            print(v);
    }
}