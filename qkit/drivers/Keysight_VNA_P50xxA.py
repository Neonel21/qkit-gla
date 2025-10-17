# Keysight P50xxA/P5004A Streamline Series USB VNA
# Joao Barbosa <j.barbosa.1@research.gla.ac.uk>, 2021
# Refactored for ModernInstrument and PrologixGPIBInstrument base classes
# Murat Baran Polat <2950172p@student.gla.ac.uk>

import logging
import numpy as np
import time
from typing import List

# Import the new base class and necessary decorators/metadata from instrument_basev2
from qkit.core.instrument_basev2 import (
    PrologixGPIBInstrument,
    QkitProperty,
    QkitFunction,
    interval_check
)

class Keysight_VNA_P50xxA(PrologixGPIBInstrument):
    """
    Driver for the Keysight P50xxA/P5004A Streamline Series USB VNA.

    This driver is designed for the modern QKIT instrument framework, inheriting
    from PrologixGPIBInstrument and using QkitProperty/QkitFunction decorators.

    Usage Example:
    --------------
    # It is recommended to use the qkit instrument factory to create an instance.
    # The timeout is set in milliseconds. For cryogenic measurements, a long
    # timeout (e.g., 60000ms) is essential.
    vna = qkit.instruments.create(
        "vna",
        "Keysight_VNA_P50xxA",
        gpib_address_str="GPIB::1::INSTR",
        ip="192.168.0.100",
        timeout=60000
    )

    # Set parameters using properties
    vna.centre_freq = 4.5e9  # 4.5 GHz
    vna.power = -10          # -10 dBm
    vna.averages = 16

    # Perform a synchronised measurement
    vna.pre_measurement()
    vna.start_measurement()
    freqs = vna.get_freqpoints()
    amp, pha = vna.get_tracedata()
    vna.post_measurement()

    Important Note:
    ---------------
    This driver uses the '*OPC?' (Operation Complete Query) command for measurement
    synchronisation. This is a blocking call that forces the script to wait until
    the VNA has finished all its sweeps and averages, making it highly reliable
    for long measurements.
    """
    def __init__(self, name: str, gpib_address_str: str, channel_id: int = 1, cw_mode: bool = False, **kwargs) -> None:
        logging.info(f"{__name__} : Initialising instrument {name}")
        
        # Call the parent PrologixGPIBInstrument's constructor.
        # It handles setting up the connection and the instrument name (`self._name`).
        super().__init__(name, gpib_address_str=gpib_address_str, **kwargs)

        self._ci = int(channel_id) # Channel ID for SCPI commands
        self.cw_mode = cw_mode     # Continuous wave mode flag
        self._edel = 0             # Electronic Delay value

        # Check for any errors that may have occurred on the instrument before we start
        self.check_errors()

        # Fetch the initial state of the most important parameters from the VNA
        self._initial_get_all_parameters()


    @QkitFunction
    def check_errors(self):
        """
        Queries the VNA's error queue and logs any pending errors.
        This is critical for debugging and ensuring commands execute successfully.
        It should be called after any command that changes the instrument's state.
        """
        while True:
            # SYSTem:ERRor[:NEXT]? queries the oldest error in the queue.
            error_string = self.ask("SYST:ERR?")
            if error_string:
                try:
                    error_code, error_message = error_string.split(',', 1)
                    error_code = int(error_code)
                    if error_code == 0:
                        # A code of 0 means "No error". The queue is clear.
                        break
                    else:
                        # Log any other error code found.
                        logging.error(f"{self._name} VNA Error: {error_code}, {error_message.strip()}")
                except ValueError:
                    # Handle cases where the VNA returns a non-standard error string.
                    logging.error(f"{self._name} received non-standard error message: {error_string}")
                    break # Avoid an infinite loop on malformed messages
            else:
                # If we get an empty response, assume the queue is clear.
                break

    # --- Properties (formerly add_parameter) ---

    @QkitProperty(
        type=int, units="",
        arg_checker=interval_check(1, None) # minval=1
    )
    def averages(self) -> int:
        """Sets or gets the number of averages for the sweep."""
        return int(self.ask(f"SENS{self._ci}:AVER:COUN?"))
    @averages.setter
    def averages(self, value: int):
        self.write(f"SENS{self._ci}:AVER:COUN {value}")
        self.check_errors()

    @QkitProperty(type=bool)
    def Average(self) -> bool: # Note: Capital 'A' as in original
        """Enables or disables averaging."""
        return bool(int(self.ask(f"SENS{self._ci}:AVER?")))
    @Average.setter
    def Average(self, value: bool):
        self.write(f"SENS{self._ci}:AVER {1 if value else 0}")
        self.check_errors()

    @QkitProperty(
        type=float, units="Hz",
        arg_checker=interval_check(1.0, 15e6) # minval=1, maxval=15e6
    )
    def bandwidth(self) -> float:
        """Sets or gets the Intermediate Frequency (IF) bandwidth."""
        return float(self.ask(f"SENS{self._ci}:BWID?"))
    @bandwidth.setter
    def bandwidth(self, value: float):
        self.write(f"SENS{self._ci}:BWID {value}")
        self.check_errors()

    @QkitProperty(
        type=float, units="Hz",
        arg_checker=interval_check(9e3, 20e9) # minval=9e3, maxval=20e9
    )
    def centre_freq(self) -> float:
        """Sets or gets the centre frequency of the sweep."""
        return float(self.ask(f"SENS{self._ci}:FREQ:CENT?"))
    @centre_freq.setter
    def centre_freq(self, value: float):
        self.write(f"SENS{self._ci}:FREQ:CENT {value}")
        self.check_errors()

    @QkitProperty(type=bool)
    def cw(self) -> bool:
        """Enables or disables Continuous Wave (CW) mode."""
        return self.cw_mode
    @cw.setter
    def cw(self, status: bool):
        if status:
            self.write(f"SENS{self._ci}:SWEEP:TYPE CW")
            self.cw_mode = True
        else:
            self.write(f"SENS{self._ci}:SWEEP:TYPE LIN")
            self.cw_mode = False
        self.check_errors()

    @QkitProperty(
        type=float, units="Hz",
        arg_checker=interval_check(10e6, 20e9) # minval=10e6, maxval=20e9
    )
    def cwfreq(self) -> float:
        """Sets or gets the Continuous Wave (CW) frequency."""
        return float(self.ask(f"SENS{self._ci}:FREQ:CW?"))
    @cwfreq.setter
    def cwfreq(self, value: float):
        if self.cw_mode:
            self.write(f"SENS{self._ci}:FREQ:CW {value}")
            self.check_errors()
        else:
            raise ValueError("VNA not in CW mode. Set .cw = True first.")

    @QkitProperty(
        type=float, units="s",
        arg_checker=interval_check(-10.0, 10.0) # minval=-10, maxval=10
    )
    def edel(self) -> float:
        """Sets or gets the electronic delay."""
        ch_sel = self.ask(f"CALC{self._ci}:PAR:SEL?").strip("\n")
        if ch_sel == '""':
            ch_ = self.ask(f"CALC{self._ci}:PAR:CAT?")
            strip_char = '"'
            self.write(f"CALC{self._ci}:PAR:SEL {ch_.strip(strip_char).split(',')[0]}")
        self._edel = float(self.ask(f"CALC{self._ci}:CORR:EDEL:TIME?"))
        return self._edel
    @edel.setter
    def edel(self, value: float):
        """Sets or gets the electronic delay."""
        ch_sel = self.ask(f"CALC{self._ci}:PAR:SEL?").strip("\n")
        if ch_sel == '""':
            ch_ = self.ask(f"CALC{self._ci}:PAR:CAT?")
            strip_char = '"'
            self.write(f"CALC{self._ci}:PAR:SEL {ch_.strip(strip_char).split(',')[0]}")
        self.write(f"CALC{self._ci}:CORR:EDEL:TIME {value}")
        self._edel = value
        self.check_errors()


    @QkitProperty(
        type=int, units="",
        arg_checker=interval_check(1, 1e5), # minval=1, maxval=1e5
        tags=["sweep"]
    )
    def nop(self) -> int:
        """Sets or gets the number of points in the sweep."""
        return int(self.ask(f"SENS{self._ci}:SWE:POIN?"))
    @nop.setter
    def nop(self, value: int):
        self.write(f"SENS{self._ci}:SWE:POIN {value}")
        self.check_errors()

    @QkitProperty(type=float, units="dBm")
    def power(self) -> float:
        """Sets or gets the output power on port 1."""
        return float(self.ask(f"SOUR{self._ci}:POW1?"))
    @power.setter
    def power(self, value: float):
        self.write(f"SOUR{self._ci}:POW1 {value}")
        self.check_errors()

    @QkitProperty(type=bool)
    def rf_output(self) -> bool:
        """Enables or disables RF output."""
        return bool(int(self.ask("OUTP?")))
    @rf_output.setter
    def rf_output(self, state: bool):
        self.write(f"OUTP {1 if state else 0}")
        self.check_errors()

    @QkitProperty(
        type=float, units="Hz",
        arg_checker=interval_check(70.0, 20e9) # minval=70, maxval=20e9
    )
    def span(self) -> float:
        """Sets or gets the frequency span of the sweep."""
        return float(self.ask(f"SENS{self._ci}:FREQ:SPAN?"))
    @span.setter
    def span(self, value: float):
        self.write(f"SENS{self._ci}:FREQ:SPAN {value}")
        self.check_errors()

    @QkitProperty(
        type=float, units="Hz",
        arg_checker=interval_check(9e3, 20e9) # minval=9e3, maxval=20e9
    )
    def startfreq(self) -> float:
        """Sets or gets the start frequency of the sweep."""
        return float(self.ask(f"SENS{self._ci}:FREQ:START?"))
    @startfreq.setter
    def startfreq(self, value: float):
        self.write(f"SENS{self._ci}:FREQ:START {value}")
        self.check_errors()

    @QkitProperty(
        type=float, units="Hz",
        arg_checker=interval_check(9e3, 20e9) # minval=9e3, maxval=20e9
    )
    def stopfreq(self) -> float:
        """Sets or gets the stop frequency of the sweep."""
        return float(self.ask(f"SENS{self._ci}:FREQ:STOP?"))
    @stopfreq.setter
    def stopfreq(self, value: float):
        self.write(f"SENS{self._ci}:FREQ:STOP {value}")
        self.check_errors()

    @QkitProperty(type=str)
    def sweepmode(self) -> str:
        """Sets or gets the sweep mode (CONT, HOLD, SING, or GRO)."""
        return self.ask(f"SENS{self._ci}:SWE:MODE?").strip()
    @sweepmode.setter
    def sweepmode(self, value: str):
        if value.upper() in ["CONT", "HOLD", "SING", "GRO"]:
            self.write(f"SENS{self._ci}:SWE:MODE {value.upper()}")
            self.check_errors()
        else:
            raise ValueError(f"Sweep mode unknown. Use: ['CONT','HOLD','SING','GRO']. Got '{value}'")

    @QkitProperty(type=float, units="s")
    def sweeptime(self) -> float:
        """Gets the sweep time. Note: This is often read-only on the VNA."""
        return float(self.ask(f"SENS{self._ci}:SWE:TIME?"))

    @QkitProperty(type=float, units="s")
    def sweeptime_averages(self) -> float:
        """Gets the total estimated sweep time including averages."""
        return self.sweeptime * self.averages


    # --- Main Functions ---

    @QkitFunction
    def reset(self):
        """Resets the VNA to its factory default state."""
        self.write("*RST")
        self.check_errors()

    @QkitFunction
    def data_format(self, value: str = "REAL32"):
        """Sets the data format for trace data retrieval."""
        if value.upper() == "ASC":
            self.write("FORM ASC,0")
        elif value.upper() == "REAL32":
            self.write("FORM REAL,32")
        elif value.upper() == "REAL64":
            self.write("FORM REAL,64")
        else:
            raise ValueError("Incorrect data format. Use: ['ASC','REAL32','REAL64']")
        self.check_errors()

    @QkitFunction
    def get_tracedata(self, format: str = "AmpPha"):
        """
        Retrieves the complex S-parameter data from the VNA and processes it.
        Returns a tuple of numpy arrays: (Amplitude, Phase) or (Real, Imaginary).
        """
        self.data_format("REAL32")
        self.write('FORM:BORD SWAP') # Byte order for binary data transfer

        # CALC:DATA? SDATA queries the raw, complex (real, imag) trace data.
        data = self.ask_for_values(f"CALC{self._ci}:DATA? SDATA")
        
        if not isinstance(data, np.ndarray):
            logging.error(f"get_tracedata received non-array data: {data}")
            raise TypeError("Expected numerical data from ask_for_values for trace data.")

        dataRe = data[::2]
        dataIm = data[1::2]

        if format.upper() == "AMPPHA":
            if self.cw_mode:
                datacomplex = np.mean(dataRe + 1j * dataIm)
                dataAmp = np.abs(datacomplex)
                dataPha = np.angle(datacomplex)
                return dataAmp, dataPha
            else:
                dataAmp = np.sqrt(dataRe**2 + dataIm**2)
                dataPha = np.arctan2(dataIm, dataRe)
                return dataAmp, dataPha
        elif format.upper() == "REALIMAG":
            return dataRe, dataIm
        else:
            raise ValueError('get_tracedata(): Format must be AmpPha or RealImag')

    @QkitFunction
    def get_freqpoints(self):
        """Retrieves the frequency points of the current sweep as a numpy array."""
        if self.cw_mode:
            return np.array([self.cwfreq])
        else:
            # CALC:DATA? FDATA queries the frequency data for the trace.
            return self.ask_for_values(f"CALC{self._ci}:DATA? FDATA")

    # --- Measurement Sequence Control ---

    @QkitFunction
    def pre_measurement(self):
        """
        Configures the VNA for a robust, triggered measurement sequence.
        This sets the sweep mode to single and enables averaging.
        """
        self.sweepmode = "SING"
        self.Average = True
        self.check_errors()

    @QkitFunction
    def start_measurement(self):
        """
        Initiates a complete measurement and waits for it to finish.
        This function clears the averaging buffer, triggers a new measurement,
        and then blocks until the VNA reports that the operation is complete.
        """
        self.write(f"SENS{self._ci}:AVER:CLE")
        self.write("INIT:IMM")
        
        # Use the Operation Complete query. This is a blocking call that makes the
        # script wait until the VNA has finished all sweeps and averages.
        self.ask("*OPC?")
        self.check_errors()

    @QkitFunction
    def post_measurement(self):
        """Resets the VNA to continuous sweep mode for live viewing after a measurement."""
        self.sweepmode = "CONT"
        self.check_errors()

    @QkitFunction
    def set_segment(self, start_freqs: List[float], stop_freqs: List[float], powers: List[float], bandwidths: List[float]):
        """
        Configures segmented sweep using start/stop frequency ranges.
        Parameters:
            start_freqs (List[float]): Start frequency of each segment (Hz).
            stop_freqs (List[float]): Stop frequency of each segment (Hz).
            powers (List[float]): Power level for each segment (dBm).
            bandwidths (List[float]): IF bandwidth for each segment (Hz).
        """
        if not (len(start_freqs) == len(stop_freqs) == len(powers) == len(bandwidths)):
            raise ValueError("All input lists must have the same length.")

        n_segments = len(start_freqs)
        self.write(f"SENS{self._ci}:SEGM:DEL:ALL") # Delete all previous segments

        for i, (f_start, f_stop, p, bw) in enumerate(zip(start_freqs, stop_freqs, powers, bandwidths)):
            seg_num = i + 1
            self.write(f"SENS{self._ci}:SEGM{seg_num}:ADD")
            self.write(f"SENS{self._ci}:SEGM{seg_num}:FREQ:STAR {f_start}")
            self.write(f"SENS{self._ci}:SEGM{seg_num}:FREQ:STOP {f_stop}")
            self.write(f"SENS{self._ci}:SEGM{seg_num}:POW {p}")
            self.write(f"SENS{self._ci}:SEGM{seg_num}:BWID {bw}")
            self.write(f"SENS{self._ci}:SEGM{seg_num}:STATE ON")
        
        self.write(f"SENS{self._ci}:SWE:TYPE SEGM") # Set sweep type to Segmented
        self.write(f"SENS{self._ci}:SEGM:COUN {n_segments}")
        self.check_errors()

    # --- Internal Helper Functions ---

    def _initial_get_all_parameters(self):
        """
        Queries the instrument for its current state upon initialisation.
        This populates the driver's properties with live values from the VNA.
        A try/except block is used to prevent a crash if a parameter is not
        available in the VNA's current state.
        """
        logging.info(f"{self._name}: Performing initial property refresh.")
        try:
            _ = self.averages
            _ = self.nop
            _ = self.power
            _ = self.startfreq
            _ = self.stopfreq
            _ = self.centre_freq
            _ = self.sweeptime
            _ = self.sweepmode
            _ = self.rf_output
            _ = self.edel
            _ = self.Average
            _ = self.bandwidth
            _ = self.cw
            _ = self.cwfreq
        except Exception as e:
            logging.warning(f"Could not get all initial parameters: {e}")


    # --- REMOVED METHODS ---
    # The following methods have been REMOVED as their functionality is now handled
    # by the PrologixGPIBInstrument base class or by the new @QkitProperty/Function system.
    # def write(self,msg): ...
    # def ask(self, msg): ...
    # def ask_for_values(self, msg, **kwargs): ...
    # def do_get_Average(self): ...
    # def do_set_Average(self, value): ...
    # ... and all other do_get_X / do_set_X methods