#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BNO055.h>
#include <utility/imumaths.h>

#include <Servo.h>

Adafruit_BNO055 bno = Adafruit_BNO055(55);

Servo servo_steering;
Servo servo_drive;

volatile int tickCount = 0;
volatile int lastMilli = 0;

const bool DEBUG = true;

const bool ENABLE_DRIVE = true;
const bool ENABLE_RPM = true;
const bool ENABLE_STEERING = true;

const bool ENABLE_STEERING_SWEEP_DEBUG = false;
const bool ENABLE_FAKE_HALL_SENSOR = false;

const int PIN_RPM_SENSOR = 4;
const int PIN_PWM_STEERING = 2;
const int PIN_PWM_DRIVE = 3;

int ledPin = 13;

int dummyHallSensorValue = 0;
int steeringValue = 0;
int driveValue = 0;

String inputString = "";         // a string to hold incoming data
boolean stringComplete = false;  // whether the string is complete

const int buttonPin = 53;
int buttonState = 0;
int oldButtonState = 0;

bool alive = true;
unsigned long lastkeepalive = 0;

void setup() {
  pinMode(buttonPin, INPUT);
  // reserve 200 bytes for the inputString:
  inputString.reserve(200);

  Serial.begin(115200);//115200

  Serial.println("CarRace IOHub Setup BEGIN");

  pinMode(ledPin, OUTPUT);
  digitalWrite(ledPin, LOW);    // sets the LED off

  // TODO Figure out Hall Effect sensor
  if (ENABLE_RPM) {
    // Set up for Hall Effect
    attachInterrupt(PIN_RPM_SENSOR, rpm_increment, FALLING);
  }

  if (ENABLE_STEERING) {
    servo_steering.attach(PIN_PWM_STEERING);
  }

  if (ENABLE_DRIVE) {
    servo_drive.attach(PIN_PWM_DRIVE);
    servo_drive.writeMicroseconds(1500);
    delay(3000);
  }

  if (ENABLE_FAKE_HALL_SENSOR) {
    Serial.println("Fake Hall Sensor mode enabled");
  }

  Serial.println("CarRace IOHub Setup END");
  servo_drive.write(90);  // reset to not moving.
  servo_steering.write(90);  // reset steering.

  buttonState = digitalRead(buttonPin);
  Serial.print("Button\t");
  Serial.println(buttonState);

  // Initialise the IMU
  if(!bno.begin(Adafruit_BNO055::OPERATION_MODE_IMUPLUS))
  {
    // There was a problem detecting the BNO055 ... check your connections
    Serial.print("Ooops, no BNO055 detected ... Check your wiring or I2C ADDR!");
    //while(1);
  }
  delay(1000);
  bno.setExtCrystalUse(true);
}

int loopCounter = 0;
void loop() {
  if (!alive) return;
  // ------------ KEEP ALIVE!!!! ------------
  // This hack will roll over once every 49 days. :(
  int now = millis();
  /*if (now > lastkeepalive + 1500) {
    while (true) {
      Serial.println("LOST KEEP ALIVE!!! KILLING EVERYTHING!!!");
      servo_drive.write(0);  // reset to not moving.
      digitalWrite(ledPin, LOW);  
      alive = false;
      return;
    }
  }*/

  imu::Quaternion quat = bno.getQuat();
  imu::Vector<3> vAcc = bno.getVector(Adafruit_BNO055::VECTOR_LINEARACCEL);
  imu::Vector<3> vGyro = bno.getVector(Adafruit_BNO055::VECTOR_GYROSCOPE);


  buttonState = digitalRead(buttonPin);

  if (oldButtonState != buttonState)
  {
    Serial.print("Button\t");
    Serial.println(buttonState);
  }
  oldButtonState = buttonState;

  loopCounter++;

  bool driveUpdated = false;
  bool steerUpdated = false;

  if (stringComplete) {
    // Two available commands:
    // D for drive, followed by int from
    // S for steering
    int commandCode = inputString[0];
    String val = inputString.substring(1);

    int readDriveValue = 0;
    int readSteerValue = 0;
    if ('D' == commandCode) {
      readDriveValue = val.toInt();
      driveUpdated = true;
      // Range is 0 to 180 (TODO ?)
      readDriveValue = constrain(readDriveValue, 0, 180);
    } else if ('S' == commandCode) {
      readSteerValue = val.toInt();
      steerUpdated = true;
      // Range is 0 to 180
      readSteerValue = constrain(readSteerValue, 0, 180);
    }
     if (driveUpdated) {
        driveValue = readDriveValue;
      }
      if (steerUpdated) {
        steeringValue = readSteerValue;
      }
 
    //Serial.println(inputString); 
    // clear the string:
    inputString = "";
    stringComplete = false;
  }

  if (ENABLE_FAKE_HALL_SENSOR) {
    if (0 == (loopCounter % 200000)) {
      // Fake speed sensor report, simple returns 2x drive value
      Serial.print("RPM:");
      Serial.println(2 * driveValue);
    }
  }

  if (ENABLE_STEERING && steerUpdated) {
    if (DEBUG) {
      Serial.print("Setting steering to ");
      Serial.println(steeringValue);
    }
    servo_steering.write(steeringValue);
  }

  if (ENABLE_DRIVE && driveUpdated) {
    if (DEBUG) {
      Serial.print("Setting drive to ");
      Serial.println(driveValue);
    }
    servo_drive.write(driveValue);
  }

  // Sweep steering from 0 to 180 and back
  if (ENABLE_STEERING_SWEEP_DEBUG && !ENABLE_STEERING) {
    int pos = 0;
    Serial.print("Servo 0");
    for (pos = 0; pos < 180; pos++) {
      servo_steering.write(pos);
      delay(15);
    }
    Serial.print(" to 180");
    for (pos = 180; pos >= 1; pos--) {
      servo_steering.write(pos);
      delay(15);
    }
    Serial.println(" back to 0");
    Serial.println(0);
  }

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
    if (inChar == '\n') {
      stringComplete = true;
      if (inputString == "keepalive\n") {
        alive = true;
        digitalWrite(ledPin, HIGH);  
        lastkeepalive = millis();
      }
    } 
  }
}

