int prev_s = -10;

const int BUTTON_PIN = 2;
const int REMOTE_BUTTON = 3;

void setup() {
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  pinMode(4, INPUT);
  pinMode(5, INPUT);
  prev_s = digitalRead(BUTTON_PIN);
  Serial.begin(38400);
}

int clamp(int x, int low, int high) {
  if (x < low) return low;
  if (x > high) return high;
  return x;
}

void loop() {
  int s = digitalRead(BUTTON_PIN);
  if (prev_s != s) {
    prev_s = s;
    Serial.println("S");
  }

  // Read the length of the pulse in microseconds
  int steering = (int)pulseIn(4, HIGH, 25000);
  int throttle = (int)pulseIn(5, HIGH, 25000);
  int remote_button = (int)pulseIn(REMOTE_BUTTON, HIGH, 25000);

  // Observed ranges:
  // - steering: Left = 1070, center = 1401, Left = 1733
  // - throttle: min = 1059, center = 1503, max = 1948

  // Deliberately use less precision to avoid 16-bit overflow
  const int loss = 5;

  // Map input to: min = 0, center = 90, max = 180
  //steering = clamp(90 + (steering - 1401) * (90 / loss) / (331 / loss), 0, 180);
  throttle = clamp(90 + (throttle - 1503) * (90 / loss) / (445 / loss), 0, 180);
  throttle = map(throttle, 180, 0, 0, 180);

  //Serial.println(steering);
  //Serial.print(" ");
  Serial.println(throttle);
  //Serial.print(" ");
  //Serial.println(remote_button);
  // delay for good luck. pulseIn should have already delayed
  // by a bit.
  delay(1);
}
