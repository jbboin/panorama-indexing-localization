function struct_json = create_json_structure(Ncol, Nrow, transformation, scanner_position_local, scanner_axis_local, room)

    camera_location = transformation(1:3,4);
    rotation = transformation(1:3,1:3);

    princ_center = [Ncol/2, Nrow/2];
    camera_k_matrix = [ princ_center(1), 0, princ_center(1);
                        0, princ_center(2), princ_center(2);
                        0, 0, 1];
                    
    toler = sum(sum(abs(eye(3,3) - rotation * rotation')));
    if toler>2e-3
        fprintf('Rotation not identity, up to tol %f\n', toler);
        assert(false);
    end

    quaternion = rotm2quat(rotation);
    
    struct_json.camera_k_matrix = camera_k_matrix;
    struct_json.camera_location = camera_location;
    struct_json.final_camera_rotation = quaternion;
    struct_json.scanner_position_local = scanner_position_local;
    struct_json.scanner_axis_local = scanner_axis_local;
    struct_json.camera_rt_matrix = transformation;
    struct_json.room = room;
    struct_json.image_width = Ncol;
    struct_json.image_height = Nrow;

