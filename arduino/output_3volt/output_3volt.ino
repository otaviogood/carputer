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


// Steering
Servo servo_steering;
const int PIN_PWM_STEERING = 2;
int steeringValue = 0;


// Drive
Servo servo_drive;
const int PIN_PWM_DRIVE = 3;
int driveValue = 0;
const double ESC_ZERO_PWM = 1500;

// Hall effect
volatile int tickCount = 0;
volatile int lastMilli = 0;
const int PIN_RPM_SENSOR = 4;


// String parsing
String inputString = "";         // a string to hold incoming data
boolean stringComplete = false;  // whether the string is complete


// Button stuff
const int BUTTON_PIN = 53;
int buttonState = 0;
int oldButtonState = 0;

void setup() 
{
  // Button setup
  pinMode(BUTTON_PIN, INPUT);

  // String setup
  inputString.reserve(200); // reserve 200 bytes for the inputString:

  Serial.begin(baudrate);//115200

  Serial.println("CarRace IOHub Setup BEGIN");
  
  attachInterrupt(PIN_RPM_SENSOR, rpm_increment, FALLING);

  servo_steering.attach(PIN_PWM_STEERING);
  servo_drive.attach(PIN_PWM_DRIVE);

  zeroESC();

  Serial.println("CarRace IOHub Setup END");
  servo_drive.write(90);  // reset to not moving.
  servo_steering.write(110);  // reset steering.

  buttonState = digitalRead(BUTTON_PIN);
  Serial.print("Button\t");
  Serial.println(buttonState);

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
  int now = millis();
  buttonState = digitalRead(BUTTON_PIN);

  if (oldButtonState != buttonState)
  {
    Serial.print("Button\t");
    Serial.println(buttonState);
  }
  oldButtonState = buttonState;

  bool driveUpdated = false;
  bool steerUpdated = false;

  if (stringComplete) 
  {
    // Two available commands:
    // D for drive, followed by int from
    // S for steering
    int commandCode = inputString[0];
    String val = inputString.substring(1);

    int readDriveValue = 0;
    int readSteerValue = 0;
    
    if ('D' == commandCode) 
    {
      readDriveValue = val.toInt();
      driveUpdated = true;
      // Range is 0 to 180 (TODO ?)
      readDriveValue = constrain(readDriveValue, 0, 180);
      driveValue = readDriveValue;
      servo_drive.write(driveValue);
    } 
    else if ('S' == commandCode) 
    {
      readSteerValue = val.toInt();
      steerUpdated = true;
      // Range is 0 to 180
      readSteerValue = constrain(readSteerValue, 0, 180);
      steeringValue = readSteerValue;
      servo_steering.write(steeringValue);
    }
    // clear the string:
    inputString = "";
    stringComplete = false;
  }

  if(!imuDead)
  {
    // Only read imu if it is alive and well
    readIMU();
  }


  if (tickCount > 0) {
    noInterrupts();
    int tempM = lastMilli;
    int tempCount = tickCount;
    tickCount = 0;
    interrupts();
    for (int i = 0; i < tempCount; i++) {
      Serial.print("Mil\t");
      Serial.println(tempM);
    }
  }
}

// TODO Figure out Hall Effect sensor
void rpm_increment() {
  lastMilli = millis();
  tickCount++;
}

void zeroESC()
{
  servo_drive.writeMicroseconds(ESC_ZERO_PWM);
  delay(3000);
}

void readIMU()
{
    /* Get the system status values (mostly for debugging purposes) */
  uint8_t system_status, self_test_results, system_error;
  system_status = self_test_results = system_error = 0;
  bno.getSystemStatus(&system_status, &self_test_results, &system_error);

  // Check the system status
  // If 255 or 1, error
  if(system_status == 255 || system_status == 1)
  {
    imuDead = true;
    return;
  }
  else
  {
    
    quat = bno.getQuat();
    vAcc = bno.getVector(Adafruit_BNO055::VECTOR_LINEARACCEL);
    vGyro = bno.getVector(Adafruit_BNO055::VECTOR_GYROSCOPE);
    Serial.print("IMU ");
    Serial.print(system_status);
    Serial.print(" ");
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
 }
/*
  SerialEvent occurs whenever a new data comes in the
 hardware serial RX.  This routine is run between each
 time loop() runs, so using delay inside loop can delay
 response.  Multiple bytes of data may be available.
 */
void serialEvent() {
  while (Serial.available()) {
    // get the new byte:
    char inChar = (char)Serial.read(); 
    // add it to the inputString:
    inputString += inChar;
    // if the incoming character is a newline, set a flag
    // so the main loop can do something about it:
    if (inChar == '\n') 
    {
      stringComplete = true;
    } 
  }
}

