#include "secrets.h"
#include <WiFiClientSecure.h>
#include <MQTTClient.h>
#include <ArduinoJson.h>
#include "WiFi.h"
#include <SDS011.h>
#include <DFRobot_DHT20.h>

// The MQTT topics that this device should publish/subscribe
#define AWS_IOT_PUBLISH_TOPIC   "esp32/pub"
#define AWS_IOT_SUBSCRIBE_TOPIC "esp32/sub"

#define uS_TO_S_FACTOR 1000000ULL
#define TIME_TO_SLEEP  1800
#define SENSOR_WARMUP  20

WiFiClientSecure net = WiFiClientSecure();
MQTTClient client = MQTTClient(256);

SDS011 sds;
DFRobot_DHT20 dht20;

RTC_DATA_ATTR int bootCount = 0;

void connectAWS() {
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  Serial.println("Connecting to Wi-Fi");

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  // Configure WiFiClientSecure to use the AWS IoT device credentials
  net.setCACert(AWS_CERT_CA);
  net.setCertificate(AWS_CERT_CRT);
  net.setPrivateKey(AWS_CERT_PRIVATE);

  // Connect to the MQTT broker on the AWS endpoint we defined earlier
  client.begin(AWS_IOT_ENDPOINT, 8883, net);

  // Create a message handler
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

  client.subscribe(AWS_IOT_SUBSCRIBE_TOPIC);

  Serial.println("AWS IoT Connected!");
}

void publishMessage() {
  StaticJsonDocument<2048> doc;
  int error;
  float p10, p25;

  error = 1;
  while (error) {
    error = sds.read(&p25,&p10);
  }

  dht20.begin();
  doc["temp"] = dht20.getTemperature();
  doc["humidity"] = dht20.getHumidity();
  
  doc["PM25"] = p25;
  doc["PM10"] = p10;
  
  char jsonBuffer[2048];
  serializeJson(doc, jsonBuffer);
  Serial.println("PM2.5: "+String(p25)+"    PM10: "+String(p10));

  client.publish(AWS_IOT_PUBLISH_TOPIC, jsonBuffer);
}

void messageHandler(String &topic, String &payload) {
  Serial.println("Incoming message: " + topic + " - " + payload);
}

void setup() {
  sds.begin(16, 17);
  Serial.begin(115200);

  ++bootCount;
  Serial.println("\nBoot number: " + String(bootCount));
  
  sds.wakeup();
  delay(SENSOR_WARMUP * 1000);

  connectAWS();
  
  publishMessage();
  client.loop();

  sds.sleep();

  Serial.println("Going to sleep now for " + String(TIME_TO_SLEEP) + "s");
  esp_sleep_enable_timer_wakeup((TIME_TO_SLEEP - SENSOR_WARMUP) * uS_TO_S_FACTOR);
  Serial.flush(); 
  esp_deep_sleep_start();
}

void loop() {
}
