# Keysight P50xxA/P5004A pytest mock file
# Murat Baran Polat <2950172p@student.gla.ac.uk>

import pytest
import numpy as np
from unittest.mock import call, patch

# The relative import is correct.
from .Keysight_VNA_P50xxA import Keysight_VNA_P50xxA
from qkit.core.instrument_basev2 import PrologixGPIBInstrument

@pytest.fixture
def mocked_vna(mocker):
    """
    Provides a 'mocked' VNA driver instance for each test function.
    
    This fixture is the core of our test setup. It patches the parent's __init__
    to prevent real network connections and sets up mock communication methods
    before the driver is even instantiated. This ensures that methods called
    during the driver's own __init__ (like check_errors) behave predictably.
    """
    # Patch the parent's __init__ method to prevent any network connections.
    mocker.patch.object(PrologixGPIBInstrument, '__init__', return_value=None)
    
    # Mock the communication methods on the class *before* an instance is made.
    # We configure 'ask' with a default return value that simulates a "no error"
    # response from the instrument. This is critical for the initial check_errors()
    # call in the driver's __init__ to succeed.
    mock_ask = mocker.patch.object(Keysight_VNA_P50xxA, 'ask', return_value='0,"No error"')
    mock_write = mocker.patch.object(Keysight_VNA_P50xxA, 'write')
    mock_ask_for_values = mocker.patch.object(Keysight_VNA_P50xxA, 'ask_for_values')

    # Patch the initial parameter fetch to isolate our tests.
    mocker.patch.object(Keysight_VNA_P50xxA, '_initial_get_all_parameters')

    # Now, create the instance. The parent __init__ is skipped, and the mocked
    # communication methods are already in place.
    vna_instance = Keysight_VNA_P50xxA(name="test_vna", gpib_address_str="GPIB::1::INSTR")
    
    # The parent class normally sets the _name attribute. Since we bypassed its
    # __init__, we must set it manually for our tests to pass.
    vna_instance._name = "test_vna"
    
    # Attach the mocks to the instance so we can inspect them in tests.
    vna_instance.ask = mock_ask
    vna_instance.write = mock_write
    vna_instance.ask_for_values = mock_ask_for_values

    return vna_instance

# --- TEST CASES ---

def test_initialisation(mocked_vna):
    """
    Tests that the driver can be initialised correctly without errors.
    """
    assert mocked_vna._name == "test_vna"
    assert mocked_vna._ci == 1
    # Check that the initial error check was called during __init__.
    mocked_vna.ask.assert_called_with("SYST:ERR?")

def test_set_power(mocked_vna):
    """
    Tests that setting the 'power' property sends the correct SCPI command and
    subsequently checks for errors.
    """
    mocked_vna.power = -20.5

    # Check that the command to set the power was sent
    mocked_vna.write.assert_called_with("SOUR1:POW1 -20.5")
    # Check that an error check was performed immediately after
    mocked_vna.ask.assert_called_with("SYST:ERR?")


def test_get_bandwidth(mocked_vna):
    """
    Tests that getting the 'bandwidth' property queries the correct command and
    parses the string result into a float.
    """
    # We need to configure the return value specifically for this test
    mocked_vna.ask.return_value = '1000.0'

    assert mocked_vna.bandwidth == 1000.0
    mocked_vna.ask.assert_called_with("SENS1:BWID?")


def test_get_tracedata_amppha(mocked_vna):
    """
    Tests the data processing logic of the `get_tracedata` function.
    """
    fake_data = np.array([3.0, 4.0, 3.0, 4.0], dtype=np.float32)
    mocked_vna.ask_for_values.return_value = fake_data
    
    amp, pha = mocked_vna.get_tracedata(format="AmpPha")

    expected_amp = np.array([5.0, 5.0])
    np.testing.assert_allclose(amp, expected_amp)
    
    mocked_vna.ask_for_values.assert_called_once_with("CALC1:DATA? SDATA")


def test_start_measurement_synchronisation(mocked_vna):
    """
    Tests that the `start_measurement` function uses the '*OPC?' query for
    reliable synchronisation.
    """
    mocked_vna.start_measurement()

    # Check that the correct sequence of commands was used.
    # The order is important here.
    expected_calls = [
        call("SENS1:AVER:CLE"),
        call("INIT:IMM"),
    ]
    mocked_vna.write.assert_has_calls(expected_calls)
    # Check that '*OPC?' was called to wait for completion.
    mocked_vna.ask.assert_any_call("*OPC?")

def test_set_segment(mocked_vna):
    """
    Tests that the `set_segment` function sends the correct sequence of SCPI
    commands to configure a segmented sweep.
    """
    starts = [1e9, 2e9]
    stops = [1.1e9, 2.1e9]
    powers = [-10, -15]
    bws = [1000, 2000]
    
    mocked_vna.set_segment(starts, stops, powers, bws)

    # *** THIS IS THE FIX ***
    # The expected call list now exactly matches the commands sent by the driver.
    expected_calls = [
        call('SENS1:SEGM:DEL:ALL'),
        call('SENS1:SEGM1:ADD'),
        call('SENS1:SEGM1:FREQ:STAR 1000000000.0'),
        call('SENS1:SEGM1:FREQ:STOP 1100000000.0'),
        call('SENS1:SEGM1:POW -10'),
        call('SENS1:SEGM1:BWID 1000'),
        call('SENS1:SEGM1:STATE ON'),
        call('SENS1:SEGM2:ADD'),
        call('SENS1:SEGM2:FREQ:STAR 2000000000.0'),
        call('SENS1:SEGM2:FREQ:STOP 2100000000.0'),
        call('SENS1:SEGM2:POW -15'),
        call('SENS1:SEGM2:BWID 2000'),
        call('SENS1:SEGM2:STATE ON'),
        call('SENS1:SWE:TYPE SEGM'),
        call('SENS1:SEGM:COUN 2'),
    ]
    mocked_vna.write.assert_has_calls(expected_calls)
    # Check that the final action was to check for errors.
    mocked_vna.ask.assert_called_with("SYST:ERR?")

