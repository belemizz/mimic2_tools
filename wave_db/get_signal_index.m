function [ signal ] = get_signal_index( info , duration )
% return signal start end frequency

freq = max([info.SamplingFrequency]);
signal.End = max([info.LengthSamples]);

if freq == 1
  signal.Start = max(1, signal.End - duration + 1);
else
  signal.Start = max(1,signal.End - floor(duration / 60) + 1);
end

end

