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
  // - steering: min = 1175, center = 1475, max = 1840
  // - throttle: min = 1010, center = 1465, max = 1975

  // Deliberately use less precision to avoid 16-bit overflow
  const int loss = 5;

  // Map input to: min = 0, center = 90, max = 180
  steering = clamp(90 + (steering - 1475) * (90 / loss) / (300 / loss), 0, 180);
  throttle = clamp(90 + (throttle - 1465) * (90 / loss) / (455 / loss), 0, 180);

  Serial.print(steering);
  Serial.print(" ");
  Serial.print(throttle);
  Serial.print(" ");
  Serial.println(remote_button);
  // delay for good luck. pulseIn should have already delayed
  // by a bit.
  delay(1);
}
