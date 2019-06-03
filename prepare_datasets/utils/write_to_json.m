function write_to_json(struct_json, json_filename)
    

jsonStr = jsonencode(struct_json);
fid = fopen(json_filename, 'w');
if fid == -1, error('Cannot create JSON file'); end
fwrite(fid, jsonStr, 'char');
fclose(fid);