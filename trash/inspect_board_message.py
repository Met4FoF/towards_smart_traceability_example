from MET4FOFDataReceiver import DataReceiver
import time
from time_series_metadata.scheme import MetaData

DR = DataReceiver("127.0.0.1", 7654)
time.sleep(5)



def simple_callback(message, description):
    print(message, description)


try:
    firstSensorId = list(DR.AllSensors.keys())[0]

    device_id = str(DR.AllSensors[firstSensorId])
    time_name = "time"
    time_unit = "\\second"
    quantity_names = []
    quantity_units = []
    misc = {
        "channel_specifics": {
            "chid": [],
            "min_scale": [],
            "max_scale": [],
            "resolution": [],
        }
    }

    channels = DR.AllSensors[firstSensorId].Description.Channels
    for i_channel in channels.keys():
        description = channels[i_channel].Description

        quantity_names.append(description["PHYSICAL_QUANTITY"])
        quantity_units.append(description["UNIT"])
        
        misc["channel_specifics"]["chid"].append(description["CHID"])
        misc["channel_specifics"]["min_scale"].append(description["MIN_SCALE"])
        misc["channel_specifics"]["max_scale"].append(description["MAX_SCALE"])
        misc["channel_specifics"]["resolution"].append(description["RESOLUTION"])

    meta = MetaData(
        device_id=device_id,
        time_name=time_name,
        time_unit=time_unit,
        quantity_names=quantity_names,
        quantity_units=quantity_units,
        misc=misc,
    )

    DR.AllSensors[firstSensorId].SetCallback(simple_callback)
    print(meta.quantities["quantity_names"])


finally:
    DR.stop()

print(meta)