#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BNO055.h>
#include <utility/imumaths.h>

#include <Servo.h>

double baudrate = 115200;

//// IMU Stuff
Adafruit_BNO055 bno = Adafruit_BNO055(55);
imu::Quaternion quat; 
imu::Vector<3> vAcc;
imu::Vector<3> vGyro;
bool imuDead = false;

void setup() 
{
  Serial.begin(baudrate);//115200

  // Initialise the IMU
  if(!bno.begin(Adafruit_BNO055::OPERATION_MODE_IMUPLUS))
  {
    // There was a problem detecting the BNO055 ... check your connections
    Serial.print("Ooops, no BNO055 detected ... Check your wiring or I2C ADDR!");
    imuDead = true;
  }
  delay(1000);
  bno.setExtCrystalUse(true);
}

void loop() 
{

  if(!imuDead)
  {
    // Only read imu if it is alive and well
    readIMU();
  }

}

void readIMU()
{
  quat = bno.getQuat();
  vAcc = bno.getVector(Adafruit_BNO055::VECTOR_LINEARACCEL);
  vGyro = bno.getVector(Adafruit_BNO055::VECTOR_GYROSCOPE);
  Serial.print("IMU ");
  Serial.print(quat.x(), 4);
  Serial.print(" ");
  Serial.print(quat.y(), 4);
  Serial.print(" ");
  Serial.print(quat.z(), 4);
  Serial.print(" ");
  Serial.print(quat.w(), 4);
  Serial.print(" ");
  Serial.print(vGyro.x(), 4);
  Serial.print(" ");
  Serial.print(vGyro.y(), 4);
  Serial.print(" ");
  Serial.print(vGyro.z(), 4);
  Serial.print(" ");
  Serial.print(vAcc.x(), 4);
  Serial.print(" ");
  Serial.print(vAcc.y(), 4);
  Serial.print(" ");
  Serial.print(vAcc.z(), 4);
  Serial.println("");
}


