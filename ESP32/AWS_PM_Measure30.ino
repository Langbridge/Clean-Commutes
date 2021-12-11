#include "secrets.h"
#include <WiFiClientSecure.h>
#include <MQTTClient.h>
#include <ArduinoJson.h>
#include "WiFi.h"
#include <SDS011.h>
#include <DFRobot_DHT20.h>

// define the MQTT topics
#define AWS_IOT_PUBLISH_TOPIC   "esp32/pub"
#define AWS_IOT_SUBSCRIBE_TOPIC "esp32/sub"

// define the conversion factor & params for device sleep
#define uS_TO_S_FACTOR 1000000ULL
#define TIME_TO_SLEEP  1800
#define SENSOR_WARMUP  20

// setup the WiFi and MQTT clients
WiFiClientSecure net = WiFiClientSecure();
MQTTClient client = MQTTClient(256);

// setup the sensors
SDS011 sds;
DFRobot_DHT20 dht20;

// initialise the boot count to zero
RTC_DATA_ATTR int bootCount = 0;

// function to connect to WiFi and AWS
void connectAWS() {
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  Serial.println("Connecting to Wi-Fi");

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  // configure WiFi to use AWS device credentials
  net.setCACert(AWS_CERT_CA);
  net.setCertificate(AWS_CERT_CRT);
  net.setPrivateKey(AWS_CERT_PRIVATE);

  // connect to the MQTT broker on the AWS endpoint
  client.begin(AWS_IOT_ENDPOINT, 8883, net);

  // create a message handler
  client.onMessage(messageHandler);

  Serial.println("Connecting to AWS IOT");

  while (!client.connect(THINGNAME)) {
    Serial.print(".");
    delay(100);
  }

  if(!client.connected()) {
    Serial.println("AWS IoT Timeout!");
    return;
  }

  // subscribe to the relevant topic
  client.subscribe(AWS_IOT_SUBSCRIBE_TOPIC);

  Serial.println("AWS IoT Connected!");
}

// function to publish messages
void publishMessage() {
  StaticJsonDocument<2048> doc;
  int error;
  float p10, p25;

  // initialise the DHT20 to provide warmup time
  dht20.begin();

  // ensure PM readings are valid
  error = 1;
  while (error) {
    error = sds.read(&p25,&p10);
  }

  // write data to the JSON doc like a Python dict
  doc["temp"] = dht20.getTemperature();
  doc["humidity"] = dht20.getHumidity();
  doc["PM25"] = p25;
  doc["PM10"] = p10;

  // serialise the JSON message
  char jsonBuffer[2048];
  serializeJson(doc, jsonBuffer);

  // output the PM values & publish the serialised message
  Serial.println("PM2.5: "+String(p25)+"    PM10: "+String(p10));
  client.publish(AWS_IOT_PUBLISH_TOPIC, jsonBuffer);
}

// function to print incoming messages
void messageHandler(String &topic, String &payload) {
  Serial.println("Incoming message: " + topic + " - " + payload);
}

// function to handle the sleep-wake cycling of the ESP32
void setup() {
  // set up communication with SDS011 & serial port
  sds.begin(16, 17);
  Serial.begin(115200);

  // increment boot count
  ++bootCount;
  Serial.println("\nBoot number: " + String(bootCount));

  // wake up & warm up the SDS011
  sds.wakeup();
  delay(SENSOR_WARMUP * 1000);

  // connect to WiFi and AWS
  connectAWS();

  // measure data & publish to AWS
  publishMessage();
  client.loop();

  // return the SDS011 to sleep
  sds.sleep();

  // send the ESP32 to deep sleep
  Serial.println("Going to sleep now for " + String(TIME_TO_SLEEP) + "s");
  esp_sleep_enable_timer_wakeup((TIME_TO_SLEEP - SENSOR_WARMUP) * uS_TO_S_FACTOR);
  Serial.flush(); 
  esp_deep_sleep_start();
}

// empty loop function, never called
void loop() {}
