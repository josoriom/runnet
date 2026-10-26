use crate::error::Error;

const NUMBER: u64 = 0;
const FIXED_64: u64 = 1;
const BYTES: u64 = 2;
const FIXED_32: u64 = 5;

pub(crate) enum FieldValue<'a> {
    Number(u64),
    Fixed64(u64),
    Bytes(&'a [u8]),
    Fixed32(u32),
}

pub(crate) struct Field<'a> {
    pub number: u64,
    pub value: FieldValue<'a>,
}

impl<'a> Field<'a> {
    pub(crate) fn as_bytes(&self) -> Result<&'a [u8], Error> {
        match self.value {
            FieldValue::Bytes(bytes) => Ok(bytes),
            _ => Err(Error::wrong_file("expected bytes")),
        }
    }

    pub(crate) fn as_text(&self) -> Result<String, Error> {
        let bytes = self.as_bytes()?;
        String::from_utf8(bytes.to_vec()).map_err(|_| Error::wrong_file("expected text"))
    }

    pub(crate) fn as_int(&self) -> Result<i64, Error> {
        match self.value {
            FieldValue::Number(number) => Ok(number as i64),
            _ => Err(Error::wrong_file("expected a number")),
        }
    }

    pub(crate) fn as_float(&self) -> Result<f32, Error> {
        match self.value {
            FieldValue::Fixed32(bits) => Ok(f32::from_bits(bits)),
            _ => Err(Error::wrong_file("expected a float")),
        }
    }

    pub(crate) fn add_ints_to(&self, output: &mut Vec<i64>) -> Result<(), Error> {
        match self.value {
            FieldValue::Number(number) => output.push(number as i64),
            FieldValue::Bytes(bytes) => {
                let mut reader = ProtoReader::new(bytes);
                while !reader.is_done() {
                    output.push(reader.read_number()? as i64);
                }
            }
            _ => return Err(Error::wrong_file("expected numbers")),
        }
        Ok(())
    }

    pub(crate) fn add_floats_to(&self, output: &mut Vec<f32>) -> Result<(), Error> {
        match self.value {
            FieldValue::Fixed32(bits) => output.push(f32::from_bits(bits)),
            FieldValue::Bytes(bytes) => {
                let (chunks, _) = bytes.as_chunks::<4>();
                output.extend(chunks.iter().map(|chunk| f32::from_le_bytes(*chunk)));
            }
            _ => return Err(Error::wrong_file("expected floats")),
        }
        Ok(())
    }

    pub(crate) fn add_doubles_to(&self, output: &mut Vec<f64>) -> Result<(), Error> {
        match self.value {
            FieldValue::Fixed64(bits) => output.push(f64::from_bits(bits)),
            FieldValue::Bytes(bytes) => {
                let (chunks, _) = bytes.as_chunks::<8>();
                output.extend(chunks.iter().map(|chunk| f64::from_le_bytes(*chunk)));
            }
            _ => return Err(Error::wrong_file("expected doubles")),
        }
        Ok(())
    }
}

pub(crate) struct ProtoReader<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> ProtoReader<'a> {
    pub(crate) fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }

    pub(crate) fn is_done(&self) -> bool {
        self.position >= self.bytes.len()
    }

    pub(crate) fn read_field(&mut self) -> Result<Field<'a>, Error> {
        let key = self.read_number()?;
        let number = key >> 3;
        let value = match key & 7 {
            NUMBER => FieldValue::Number(self.read_number()?),
            FIXED_64 => FieldValue::Fixed64(u64::from_le_bytes(self.read_array()?)),
            BYTES => {
                let length = self.read_number()? as usize;
                FieldValue::Bytes(self.read_bytes(length)?)
            }
            FIXED_32 => FieldValue::Fixed32(u32::from_le_bytes(self.read_array()?)),
            _ => return Err(Error::wrong_file("unknown field type")),
        };
        Ok(Field { number, value })
    }

    fn read_number(&mut self) -> Result<u64, Error> {
        let mut number = 0_u64;
        for shift in (0..64).step_by(7) {
            let byte = *self
                .bytes
                .get(self.position)
                .ok_or_else(|| Error::wrong_file("the file ends too early"))?;
            self.position += 1;
            number |= u64::from(byte & 0x7f) << shift;
            if byte < 0x80 {
                return Ok(number);
            }
        }
        Err(Error::wrong_file("a number is too long"))
    }

    fn read_bytes(&mut self, length: usize) -> Result<&'a [u8], Error> {
        let end = self
            .position
            .checked_add(length)
            .filter(|end| *end <= self.bytes.len())
            .ok_or_else(|| Error::wrong_file("the file ends too early"))?;
        let bytes = &self.bytes[self.position..end];
        self.position = end;
        Ok(bytes)
    }

    fn read_array<const SIZE: usize>(&mut self) -> Result<[u8; SIZE], Error> {
        let mut array = [0_u8; SIZE];
        array.copy_from_slice(self.read_bytes(SIZE)?);
        Ok(array)
    }
}
