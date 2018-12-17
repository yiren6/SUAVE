## @ingroup Surrogate
# neural_network_surrogate_functions.py

# Created: Nov 2018, J. Smart
# Modified:


# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Core import Units
from SUAVE.Methods.Weights.Buildups.Electric_Stopped_Rotor.empty import empty

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import adam
from keras.utils import to_categorical

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# Build Dataset
# ----------------------------------------------------------------------

def build_dataset(N=1000):
    
# ----------------------------------------------------------------------
# Vehicle Level Properties
# ----------------------------------------------------------------------

    MTOW_Space      = np.linspace(2000., 20000., 200)
    battery_Space   = np.linspace(10., 1000., 100)
    payload_Space   = np.linspace(10., 5000., 500)

# ----------------------------------------------------------------------
# Fuselage Properties
# ----------------------------------------------------------------------

    fLength_Space   = np.linspace(3., 10., 20)
    fWidth_Space    = np.linspace(2., 10., 20)
    fHeight_Space   = np.linspace(1., 3., 5)

# ----------------------------------------------------------------------
# Wing Properties
# ----------------------------------------------------------------------

    span_Space      = np.linspace(0.1, 20., 200)
    chord_Space     = np.linspace(0.1, 5., 50)
    tc_Space        = np.linspace(0.01, 0.3, 10)
    wf_Space        = np.linspace(0., 0.25, 10)

# ----------------------------------------------------------------------
# Propeller Properties
# ----------------------------------------------------------------------

    liftCount_Space     = np.linspace(2, 5, 4)
    liftBlade_Space     = np.linspace(2, 6, 5)
    thrustCount_Space   = np.linspace(1, 4, 4)
    thrustBlade_Space   = np.linspace(2, 6, 5)
    tipRadius_Space     = np.linspace(0.1, 5., 50)

# ----------------------------------------------------------------------
# Create Specification List
# ----------------------------------------------------------------------

    numDatapoints = N
    results = []

    for i in range(numDatapoints):

        print('Generating Point: {}'.format(i))

        # --------------------------------------------------------------
        # Unpack Inputs
        # --------------------------------------------------------------

        MTOW = MTOW_Space[np.random.randint(len(MTOW_Space))]
        batmass = battery_Space[np.random.randint(len(battery_Space))]
        payload = payload_Space[np.random.randint(len(payload_Space))]
        fLength = fLength_Space[np.random.randint(len(fLength_Space))]
        fWidth = fWidth_Space[np.random.randint(len(fWidth_Space))]
        fHeight = fHeight_Space[np.random.randint(len(fHeight_Space))]
        span = span_Space[np.random.randint(len(span_Space))]
        chord = chord_Space[np.random.randint(len(chord_Space))]
        tc = tc_Space[np.random.randint(len(tc_Space))]
        wf = wf_Space[np.random.randint(len(wf_Space))]
        liftCount = int(liftCount_Space[np.random.randint(len(liftCount_Space))])
        liftBlade = liftBlade_Space[np.random.randint(len(liftBlade_Space))]
        thrustCount = int(thrustCount_Space[np.random.randint(len(thrustCount_Space))])
        thrustBlade = thrustBlade_Space[np.random.randint(len(thrustBlade_Space))]
        tipRadius = tipRadius_Space[np.random.randint(len(tipRadius_Space))]

        vehicle = SUAVE.Vehicle()
        vehicle.tag = "Attribute Holder"

        vehicle.mass_properties.max_takeoff = MTOW

        net = SUAVE.Components.Energy.Networks.Lift_Forward_Propulsor()
        net.number_of_engines_lift = liftCount
        net.number_of_engines_forward = thrustCount
        net.number_of_engines = liftCount+thrustCount

        payloadObject = SUAVE.Components.Energy.Peripherals.Payload()
        payloadObject.mass_properties.mass = payload * Units.kg
        net.payload = payloadObject

        bat = SUAVE.Components.Energy.Storages.Batteries.Constant_Mass.Lithium_Ion()
        bat.mass_properties.mass = batmass
        net.battery = bat

        prop_lift = SUAVE.Components.Energy.Converters.Propeller()
        prop_lift.prop_attributes.tip_radius = tipRadius
        prop_lift.prop_attributes.number_blades = liftBlade
        prop_lift.tag = "Forward_Prop"
        net.propeller_lift = prop_lift

        prop_fwd = SUAVE.Components.Energy.Converters.Propeller()
        prop_fwd.prop_attributes.tip_radius = tipRadius
        prop_fwd.prop_attributes.number_blades = thrustBlade
        prop_fwd.tag = "Thrust_Prop"
        net.propeller_forward = prop_fwd

        vehicle.append_component(net)

        wing = SUAVE.Components.Wings.Main_Wing()
        wing.tag = 'main_wing'
        wing.spans.projected = span
        wing.chords.mean_aerodynamic = chord
        wing.thickness_to_chord = tc
        wing.winglet_fraction = wf
        wing.areas.reference = span * chord
        wing.motor_spanwise_locations = np.linspace(0., 1., liftCount)

        vehicle.append_component(wing)

        sec_wing = SUAVE.Components.Wings.Wing()
        sec_wing.tag =  'secondary_wing'
        sec_wing.spans.projected = span
        sec_wing.chords.mean_aerodynamic = chord
        sec_wing.thickness_to_chord = tc
        sec_wing.winglet_fraction = wf
        sec_wing.areas.reference = span * chord
        sec_wing.motor_spanwise_locations = np.linspace(0., 1., liftCount)
        vehicle.append_component(sec_wing)

        fuselage = SUAVE.Components.Fuselages.Fuselage()
        fuselage.lengths.total = fLength
        fuselage.width = fWidth
        fuselage.heights.maximum = fHeight
        vehicle.append_component(fuselage)

        response = empty(vehicle).empty

        results.append([MTOW, batmass, payload,
                        fLength, fWidth, fHeight,
                        span, chord, tc, wf,
                        liftCount, liftBlade, thrustCount, thrustBlade, tipRadius,
                        response])


    storage = pd.DataFrame(results, columns=['MTOW', 'Bat. Mass', 'Payload',
                                'F-Length', 'F-Width', 'F-Height',
                                'Wingspan', 'Chord', 'T-C', 'WL Frac.',
                                'L-Rotors', 'L-Blades', 'T-Rotors', 'T-Blades', 'R-Radius',
                                'EVW'])
    fileroot = 'C:\\Users\\Jordan Smart\\Documents\\Classwork\\CS230 - Deep Learning\\Final Project\\Datasets\\N'
    storage.to_csv(fileroot+str(numDatapoints)+'.csv')

# ----------------------------------------------------------------------
# Basic Reference Model
# ----------------------------------------------------------------------

def build_base_model():

    HU = 128

    model = Sequential()
    model.add(Dense(HU, input_shape=(15,)))
    model.add(Dense(HU))
    model.add(Dense(HU))
    model.add(Dense(HU))
    model.add(Dense(HU))
    model.add(Dense(HU))
    model.add(Dense(1, activation='relu'))

    opt = adam(lr=0.005)

    model.compile(optimizer=opt,
                  loss = 'mean_squared_error',
                  metrics=['accuracy'])

    return model

def import_data(N=10000, trainFraction = 0.9, categorical=False):

    fileroot = 'C:\\Users\\Jordan Smart\\Documents\\Classwork\\CS230 - Deep Learning\\Final Project\\Datasets\\N'
    data = pd.read_csv(fileroot + str(N)+'.csv')

    dataArray = data.values
    dataArray = dataArray[:,1:]

    split = int(N*trainFraction)

    x_train = dataArray[0:split, 0:15]
    x_test  = dataArray[split:N, 0:15]

    y_train = dataArray[0:split, 15]
    y_test  = dataArray[split:N, 15]

    if categorical:
        y_train = (y_train/100).astype(int)
        y_test  = (y_test/100).astype(int)

        y_train[y_train>100] = 100
        y_test[y_test>100] = 100

        y_train = to_categorical(y_train, num_classes=101)
        y_test = to_categorical(y_test, num_classes=101)

    return x_train, x_test, y_train, y_test

# ----------------------------------------------------------------------
# Main Function
# ----------------------------------------------------------------------

def main():

    #build_dataset(N=100000)

    x_train, x_test, y_train, y_test = import_data(N=100000, trainFraction=0.95, categorical=False)

    model = build_base_model()

    model.fit(x_train, y_train, epochs = 500, batch_size=1024, verbose=2)

    trainScore = model.evaluate(x_train, y_train, batch_size=1024)
    testScore = model.evaluate(x_test, y_test, batch_size=1024)

    print('Training Acc.: {}'.format(trainScore))
    print('Testing Acc.: {}'.format(testScore))

    model.save('C:\\Users\\Jordan Smart\\Documents\\Classwork\\CS230 - Deep Learning\\Final Project\\Models\\Cat100000.h5')

if __name__ == "__main__":
    main()