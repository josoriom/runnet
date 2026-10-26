use crate::{model_file::get_half, proto_reader::ProtoReader};

#[test]
fn repeated_numbers_are_read_packed_and_unpacked() {
    let unpacked = [0x08, 0x03, 0x08, 0x96, 0x01];
    let packed = [0x0a, 0x03, 0x03, 0x96, 0x01];
    let negative = [
        0x08, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x01,
    ];
    let mut numbers = Vec::new();

    for bytes in [&unpacked[..], &packed[..], &negative[..]] {
        let mut reader = ProtoReader::new(bytes);
        while !reader.is_done() {
            reader
                .read_field()
                .expect("field")
                .add_ints_to(&mut numbers)
                .expect("numbers");
        }
    }

    assert_eq!(numbers, vec![3, 150, 3, 150, -1]);
}

#[test]
fn half_floats_keep_their_special_values() {
    assert_eq!(get_half(0x3c00), 1.0);
    assert_eq!(get_half(0xc000), -2.0);
    assert_eq!(get_half(0x0001), 2_f32.powi(-24));
    assert_eq!(get_half(0x7c00), f32::INFINITY);
    assert_eq!(get_half(0xfc00), f32::NEG_INFINITY);
    assert!(get_half(0x7e00).is_nan());
}
