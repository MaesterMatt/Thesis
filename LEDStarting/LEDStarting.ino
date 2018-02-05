#include "DualVNH5019MotorShield.h"

DualVNH5019MotorShield md;

int immap = 0;
int leftMotor;
int rightMotor;
int led = 27;
int m1_0 = 3;
int m1_1 = 4;
int m2_1 = 5;
int m2_0 = 6;
int m1_en = 2;
int m2_en = 7;

void stopIfFault(){
  if (md.getM1Fault() | md.getM2Fault())
  {
    md.setM1Speed(0);
    md.setM2Speed(0);
    while(1);
  }
}
void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);

  md.init(); //start the dual vnh5019 motor shield
}

void loop() {
  // put your main code here, to run repeatedly:
  if (Serial.available() > 0){
    immap = Serial.read();
    if (immap == 0){
      md.setM1Speed(0);
      md.setM2Speed(0);
      stopIfFault();
    }
    else{
      md.setM1Speed(map(immap, 0, 256, 100, 256));
      md.setM2Speed(map(immap, 0, 256, 240, 100));      
    }
  }
}
