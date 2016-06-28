/**************************************
 * Memorial University of Newfoundland
 * SWASP/MUCEP Job Placement
 * Wave Dynamics Lab Pump Automation
 * Firmware for Arduino Leonardo
 * Designed to run MasterFlex I/P Pumps
 * Chartered by Dr. James Munroe
 * Contact: jmunroe@mun.ca
 * Written by Dean Massecar
 * Contact: dam671@mun.ca
 * Alt Contact: dean.massecar@gmail.com
 * Created: 2013-06-04
 * Last updated: 2013-06-14
 **************************************/

//pin settings
const int P1_INPUT_PIN = 10; //to pump
const int P1_OUTPUT_PIN = A0; //from pump
const int P2_INPUT_PIN = 11;
const int P2_OUTPUT_PIN = A1;
const int P1_STARTSTOP = 2;
const int P2_STARTSTOP = 4;
const int DEBUG_LED = 13;

const int P1_TOG_DIR = 7;
const int P2_TOG_DIR = 8;


//calibration settings
const int PWM_MAX_VALUE = 255;
const int PWM_MIN_VALUE = 0;

int pump1State = -1;
int pump2State = -1;
int pump1Speed = -1;
int pump2Speed = -1;

int stateCounter = 5;

int getPumpState(char pump)
{  //prints current pump state, -1 if error
  if(pump=='1')
  {
    return pump1State;

  }
  else if(pump=='2')
  {
    return pump2State;
  }
  else
  {
    return -1;
  }

}

int getPumpSpeed(char pump)
{ //prints curent pump speed, -1 if error
  if(pump == '1')
  {
    return pump1Speed;
  }
  else if(pump == '2')
  {
    return pump2Speed;
  }
  else
  {
    return -1;
  }
}


boolean setPumpValue(char pump, int pinSetting)
{ //sets the analog write on specified pump pin

  int inputPin = -1;
  if(pump == '1')
  {
    pump1Speed = pinSetting;
    inputPin = P1_INPUT_PIN;
  }
  else if(pump == '2')
  {
    pump2Speed = pinSetting;
    inputPin = P2_INPUT_PIN;
  }
  else
  {
    return false;
  }
  pinSetting = constrain(pinSetting, 0, 255);
  analogWrite(inputPin, pinSetting);
  return true;
}

float getPumpValue(int outputPin)
{
  //reads and converts from 0-1023 units, to 0-5V, or 0.0049V/unit
  //however, current is being returned, therefore drop over resistor is measured
  //R=120 ohms. V=IR. Therefore, I=V/R
  //returns pumpValue as the voltage
  float pumpValue = analogRead(outputPin) * 0.0049;
  pumpValue /= 120.0;
  pumpValue *= 1000.0; //brings to mA
  return pumpValue;
}

void togDir(int togglePin, boolean clockwise)
{ //if clockwise, boolean = true, transistor closed/open?
  if(clockwise)
  {
    digitalWrite(togglePin, HIGH);
  }
  else
  {
    digitalWrite(togglePin, LOW);
  }
}

boolean startStop(char pump, int pinSetting)
{ 
  int pinNum = -1;
  if(pump == '1')
  {
    pinNum = P1_STARTSTOP;
    pump1State = pinSetting;
  }
  else if(pump == '2')
  {
    pinNum = P2_STARTSTOP;
    pump2State = pinSetting;
  }
  else
  {
    return false;
  }

  if(pinSetting == 1)
  {
    digitalWrite(pinNum, HIGH);
    //digitalWrite(DEBUG_LED, HIGH);
  }
  else if(pinSetting == 0)
  {
    digitalWrite(pinNum, LOW);
    //digitalWrite(DEBUG_LED, LOW);
  }
  else
  {
    return false;
  }

  return true;
}

void setup()
{
  Serial.begin(9600);
  while(!Serial)
  {
    ; //required for Leonardo. Program will not continue
    //until USB serial communications have been established.
  }

  pinMode(P1_INPUT_PIN, OUTPUT);
  pinMode(P2_INPUT_PIN, OUTPUT);
  analogReference(DEFAULT);
  pinMode(P1_STARTSTOP, OUTPUT);
  pinMode(P2_STARTSTOP, OUTPUT);
  pinMode(P1_TOG_DIR, OUTPUT);
  pinMode(P1_TOG_DIR, OUTPUT);
  pinMode(DEBUG_LED, OUTPUT);

  //should ideally prevent the pumps from running at power on,
  //however, this does not seem to be the case.
  //ideally, at power up, the pump START/STOP pins should be
  //toggled HIGH then LOW, while the pumps are not in active use,
  //before actually using the pumps.
  //this will be included with the client program, and obviously
  //included in the actualy documentation for the system.
  
  startStop('1', 0);
  startStop('2', 0);
  setPumpValue('1', 0);
  setPumpValue('2', 0);
  
  //delay(1000);
  //delay(1000);
  //delay(1000);
  //Serial.print("READY\n");
  delay(50);
}

void loop()
{
  /*
  input formatting <S/G><P/T><1/2>[value] 
   which means <Set/Get><sPeed/sTate><pump1/pump2>[value for set]
   S = set
   G = get
   P = speed
   T = state
   pump = pump number [1, 2]
   value = value
   */

  //serialInput[] = "";
  //array needs to be blanked before each loop
  char serialInput[65] = {'\0'};
  //serial buffer is 64 bytes, therefore max length, char array requires n+1
  //for null character at end. Max command length = 6, +1 for null
  int numBytes=Serial.readBytesUntil('\n', serialInput, 6); //default timeout = 1000ms
  /*Serial.print(numBytes);
  Serial.print(" of ");
  Serial.print(serialInput);
  Serial.print("\n");*/
  
  if(numBytes > 0)
  {

    /*
    //for debug
    for(int i=0; i<10;i++)
    {
      Serial.print(serialInput[i]);
    }
    Serial.print("\n");
    //for debug
    */

    char dir = serialInput[0];
    char func = serialInput[1];
    char pump = serialInput[2];
    String str = "   ";
    str.setCharAt(0, serialInput[3]);
    str.setCharAt(1, serialInput[4]);
    str.setCharAt(2, serialInput[5]);
    int stateSetting = str.toInt();
    
    switch(dir)
    {

    case 'G': //get functions
      {
        switch(func)
        {
        case 'P': //get speed
          {
            if(pump == '1' || pump == '2')
            { 
              int state = getPumpSpeed(pump);
              Serial.print(state);
              //Serial.print(" OK");
              Serial.print("\n");
            }
            else
            {
              Serial.print("ERROR\n");
            }

            break;
          }
          /******/
        case 'T': //get state
          {
            if(pump == '1' || pump == '2')
            { 
              int state = getPumpState(pump);
              Serial.print(state);
              //Serial.print(" OK");
              Serial.print("\n");
            }
            else
            {
              Serial.print("ERROR\n");
            }
            break;
          }
        default:
          {
            Serial.print("ERROR\n");
            break;
          }
        }
        break;
      }
      /***************************************/
    case 'S': //set functions
      {
        switch(func)
        {
        case 'P': //set speed
          {
            if(pump == '1' || pump == '2')
            {
              stateSetting = constrain(stateSetting, 0, 255);
              setPumpValue(pump, stateSetting);
              Serial.print("OK");
              //Serial.print(stateSetting);
              Serial.print("\n");
            }
            else
            {
              Serial.print("ERROR\n");
            }
            break;
          }
          /*****/
        case 'T': //set state
          {
            if(pump == '1' || pump == '2')
            {
              stateSetting = constrain(stateSetting, 0, 1);
              startStop(pump, stateSetting);
              Serial.print("OK");
              //Serial.print(stateSetting);
              Serial.print("\n");
            }
            else
            {
              Serial.print("ERROR\n");
            }
            break;
          }
        default:
          {
            Serial.print("ERROR\n");
            break;
          }
        }
        break;
      }

    default:
      {
        Serial.print("ERROR\n"); //something is causing the final error to trigger
        //but is only triggering on SET, and not GET
        break; 
      }
    }
  }
}


