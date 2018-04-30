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
  Serial.begin(115200);

  md.init(); //start the dual vnh5019 motor shield
}
int lspeed = 0;
int rspeed = 0;
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
      leftMotor = (immap & 0xF0)>>4;
      rightMotor = immap & 0x0F;
      md.setM1Speed(map(leftMotor, 0, 15, 0, 160));
      md.setM2Speed(map(rightMotor, 0, 15, 0, 160)); 
    }
  }
  else{
    delay(60);
    md.setM1Speed(0);
    md.setM2Speed(0);
    stopIfFault();
  }
}
