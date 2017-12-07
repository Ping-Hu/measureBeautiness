function [result] = findCaID( caNumber, caAllIndex, cid )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

for m = 1:caNumber
    if caAllIndex(m).categoryID == int2str(cid)
        result = caAllIndex(m).categoryID;
        
    end
end
 
