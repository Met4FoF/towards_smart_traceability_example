from dataPlayer import SensorDataPlayer
import time


# player = SensorDataPlayer('data/Met4FOF_mpu9250_Z_Acc_10_hz_250_hz_6rep.dump')
#player = SensorDataPlayer('data/200623_MPU_9250_SN_12_X_Achse_3_COLAREF/200623_MPU_9250_SN_12_X_Achse_3_COLAREF.dump')
player = SensorDataPlayer('data/200623_MPU_9250_SN_12_X_Achse_3_COLAREF/trimmed.dump')


try:
    while True:
        time.sleep(0.5)

except KeyboardInterrupt:
    print("Shutting down player")
    player.stop()
