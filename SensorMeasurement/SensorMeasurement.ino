//Feel free to use this code.

//Please be respectful by acknowledging the author in the code if you use or modify it.

//Author: Bruce Allen

//Date: 23/07/09
#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BNO055.h>
#include <utility/imumaths.h>
#include <LIDARLite.h>

Adafruit_BNO055 bno = Adafruit_BNO055(55);
LIDARLite lidarLite;
int cal_cnt = 0;

//Digital pin 7 for reading in the pulse width from the MaxSonar device.

//This variable is a constant because the pin will not change throughout execution of this code.

const int pwPin1 = 44;
const int pwPin2 = 45;



//variables needed to store values

long pulse1, inches1;
long pulse2, inches2;
int sensorVal[] = {0, 0, 0};
int arraysize = 9;
int rangevalue1[] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
int rangevalue2[] = {0, 0, 0, 0, 0, 0, 0, 0, 0};

void setup()

{
  Serial.begin(9600);
  pinMode(pwPin1, INPUT);
  pinMode(pwPin2, INPUT);
  lidarLite.begin(0, true);
  lidarLite.configure(0);
  
  if(!bno.begin())
  {
    Serial.println("Ooops, no BNO detected");
    delay(1000);
  }
  bno.setExtCrystalUse(true);
}
void loop()
{
  sensors_event_t event;
  bno.getEvent(&event);
  sensorVal[2] = int(event.orientation.x);
  if(sensorVal[2] > 180)
    sensorVal[2] = sensorVal[2] - 360;
  if(cal_cnt == 0){
    sensorVal[0] = lidarLite.distance();
  }
  else{
    sensorVal[0] = lidarLite.distance(false);
  }
  cal_cnt++;
  cal_cnt = cal_cnt%100;
  Serial.print(sensorVal[0]); //Right sonar
  Serial.print(',');
  Serial.print(sensorVal[1]); //Left sonar
  Serial.print(',');
  Serial.println(sensorVal[2]); //IMU
  delay(10);
}
