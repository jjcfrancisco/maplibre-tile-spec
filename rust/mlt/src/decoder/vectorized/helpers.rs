use bytes::Buf;
use zigzag::ZigZag;

use crate::decoder::tracked_bytes::TrackedBytes;
use crate::decoder::varint;
use crate::metadata::stream::StreamMetadata;
use crate::metadata::stream_encoding::{LogicalLevelTechnique, PhysicalLevelTechnique};
use crate::vector::types::VectorType;
use crate::{MltError, MltResult};

/// Decode ([`ZigZag`] + delta) for Vec2s
// TODO: The encoded process is (delta + ZigZag) for each component
pub fn decode_componentwise_delta_vec2s<T: ZigZag>(data: &[T::UInt]) -> Result<Vec<T>, MltError> {
    let len = data.len();
    if len < 2 {
        return Err(MltError::MinLength {
            ctx: "vec2 delta stream",
            min: 2,
            got: len,
        });
    }
    if len % 2 != 0 {
        return Err(MltError::InvalidValueMultiple {
            ctx: "vec2 delta stream length",
            multiple_of: 2,
            got: len,
        });
    }

    let mut result = Vec::with_capacity(len);
    result.push(T::decode(data[0]));
    result.push(T::decode(data[1]));

    for i in (2..len).step_by(2) {
        result.push(T::decode(data[i]) + result[i - 2]);
        result.push(T::decode(data[i + 1]) + result[i - 1]);
    }

    Ok(result)
}

pub fn get_vector_type_int_stream(metadata: &StreamMetadata) -> VectorType {
    let tech1 = metadata.logical.technique1;
    let tech2 = metadata.logical.technique2;
    let runs = metadata.rle.as_ref().map(|r| r.runs);
    let n = metadata.num_values as usize;

    match (tech1, tech2, runs, n) {
        // L1 == RLE → runs == 1 → CONST; else FLAT
        (Some(LogicalLevelTechnique::Rle), _, Some(1), _) => VectorType::Const,
        (Some(LogicalLevelTechnique::Rle), _, Some(_), _) => VectorType::Flat,
        // L1 == DELTA && L2 == RLE && runs in {1,2} → SEQUENCE
        (Some(LogicalLevelTechnique::Delta), Some(LogicalLevelTechnique::Rle), Some(r), _)
            if r == 1 || r == 2 =>
        {
            VectorType::Sequence
        }
        // num_values == 1 → CONST; else FLAT
        (_, _, _, 1) => VectorType::Const,
        _ => VectorType::Flat,
    }

/// Decode a physical level technique.
fn decode_physical_level_technique(
    data: &mut TrackedBytes,
    metadata: &StreamMetadata,
) -> MltResult<Vec<u32>> {
    match metadata.physical.technique {
        PhysicalLevelTechnique::Varint => varint::decode::<u32>(data, metadata.num_values as usize),
        PhysicalLevelTechnique::None => {
            let byte_length = metadata.byte_length as usize;
            let mut values = Vec::with_capacity(byte_length / 4);

            // Read the raw bytes directly from the TrackedBytes
            for _ in 0..(byte_length / 4) {
                if data.remaining() < 4 {
                    return Err(MltError::InsufficientData);
                }
                let value = data.get_u32_le(); // Read u32 in little-endian format
                values.push(value);
            }

            Ok(values)
        }
        _ => Err(MltError::UnsupportedPhysicalTechnique(
            metadata.physical.technique,
        )),
    }

/// Decode a constant integer stream.
pub fn decode_const_int_stream(
    data: &mut TrackedBytes,
    metadata: &StreamMetadata,
    is_signed: bool,
) -> MltResult<i32> {
    let values = decode_physical_level_technique(data, metadata)?;

    if values.len() == 1 {
        let value = values.first().ok_or(MltError::InsufficientData)?;

        let result = if is_signed {
            ZigZag::decode(*value)
        } else {
            *value as i32
        };

        return Ok(result);
    }

    // Handle RLE case
    let result = if is_signed {
        let value = values.get(1).ok_or(MltError::InsufficientData)?;
        ZigZag::decode(*value)
    } else {
        *values.get(1).ok_or(MltError::InsufficientData)? as i32
    };

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metadata::stream::{Rle, StreamMetadata};
    use crate::metadata::stream_encoding::{
        Logical, LogicalLevelTechnique, LogicalStreamType, Physical, PhysicalLevelTechnique,
        PhysicalStreamType,
    };

    #[test]
    fn test_decode_componentwise_delta_vec2s() {
        // original Vec2s: [(3, 5), (7, 6), (12, 4)]
        // delta:          [3, 5, 4, 1, 5, -2]
        // ZigZag:         [6, 10, 8, 2, 10, 3]
        let encoded_from_positives: Vec<u32> = vec![6, 10, 8, 2, 10, 3];
        let decoded = decode_componentwise_delta_vec2s::<i32>(&encoded_from_positives).unwrap();
        assert_eq!(decoded, vec![3, 5, 7, 6, 12, 4]);

        // original Vec2s: [(3, 5), (-1, 6), (4, -4)]
        // delta:          [3, 5, -4, 1, 5, -10]
        // ZigZag:         [6, 10, 7, 2, 10, 19]
        let encoded_from_negatives: Vec<u32> = vec![6, 10, 7, 2, 10, 19];
        let decoded = decode_componentwise_delta_vec2s::<i32>(&encoded_from_negatives).unwrap();
        assert_eq!(decoded, vec![3, 5, -1, 6, 4, -4]);
    }

    fn generate_metadata(
        t1: LogicalLevelTechnique,
        t2: LogicalLevelTechnique,
        runs: Option<u32>,
        num_values: u32,
    ) -> StreamMetadata {
        StreamMetadata {
            logical: Logical::new(Some(LogicalStreamType::Dictionary(None)), t1, t2),
            physical: Physical::new(PhysicalStreamType::Present, PhysicalLevelTechnique::Varint),
            num_values,
            byte_length: 0,
            morton: None,
            rle: runs.map(|r| Rle {
                runs: r,
                num_rle_values: r * 2,
            }),
        }
    }

    #[test]
    fn table_driven_vector_type_int_stream() {
        let cases = vec![
            (
                "RLE runs = 1 → CONST",
                generate_metadata(
                    LogicalLevelTechnique::Rle,
                    LogicalLevelTechnique::Delta,
                    Some(1),
                    10,
                ),
                VectorType::Const,
            ),
            (
                "Delta + RLE runs = 2 → SEQUENCE",
                generate_metadata(
                    LogicalLevelTechnique::Delta,
                    LogicalLevelTechnique::Rle,
                    Some(2),
                    8,
                ),
                VectorType::Sequence,
            ),
            (
                "Fallback: num_values == 1 → CONST",
                generate_metadata(
                    LogicalLevelTechnique::Delta,
                    LogicalLevelTechnique::Delta,
                    None,
                    1,
                ),
                VectorType::Const,
            ),
            (
                "Default: no special case, num_values > 1 → FLAT",
                generate_metadata(
                    LogicalLevelTechnique::Delta,
                    LogicalLevelTechnique::Delta,
                    None,
                    5,
                ),
                VectorType::Flat,
            ),
        ];

        for (desc, meta, expected) in cases {
            let vt = get_vector_type_int_stream(&meta);
            assert_eq!(vt, expected, "case failed: {desc}");
        }
    }

    use bytes::Bytes;

    #[test]
    fn test_decode_physical_level_technique_varint() {
        let bytes: Bytes = Bytes::from(vec![
            0x01, // 1
            0xAC, 0x02, // 300
            0xD0, 0x86, 0x03, // 50000
        ]);
        let mut tile: TrackedBytes = bytes.into();
        let metadata = StreamMetadata {
            logical: Logical::new(
                Some(LogicalStreamType::Dictionary(None)),
                LogicalLevelTechnique::None,
                LogicalLevelTechnique::None,
            ),
            physical: Physical::new(PhysicalStreamType::Present, PhysicalLevelTechnique::Varint),
            num_values: 3,
            byte_length: 0,
            morton: None,
            rle: None,
        };

        let decoded = decode_physical_level_technique(&mut tile, &metadata).unwrap();
        assert_eq!(decoded, vec![1, 300, 50000]);
    }

    #[test]
    fn test_decode_physical_level_technique_none() {
        let bytes: Bytes = Bytes::from(vec![
            // 1 in little-endian
            0x01, 0x00, 0x00, 0x00, // 300 in little-endian
            0x2C, 0x01, 0x00, 0x00, // 50000 in little-endian
            0x50, 0xC3, 0x00, 0x00,
        ]);

        let mut tile: TrackedBytes = bytes.into();
        let metadata = StreamMetadata {
            logical: Logical::new(
                Some(LogicalStreamType::Dictionary(None)),
                LogicalLevelTechnique::None,
                LogicalLevelTechnique::None,
            ),
            physical: Physical::new(PhysicalStreamType::Present, PhysicalLevelTechnique::None),
            num_values: 3,
            byte_length: 12, // 3 values * 4 bytes each
            morton: None,
            rle: None,
        };

        let decoded = decode_physical_level_technique(&mut tile, &metadata).unwrap();
        assert_eq!(decoded, vec![1, 300, 50000]);
    }

    #[test]
    fn test_decode_physical_level_technique_none_empty() {
        let bytes: Bytes = Bytes::from(vec![]);
        let mut tile: TrackedBytes = bytes.into();

        let metadata = StreamMetadata {
            logical: Logical::new(
                Some(LogicalStreamType::Dictionary(None)),
                LogicalLevelTechnique::None,
                LogicalLevelTechnique::None,
            ),
            physical: Physical::new(PhysicalStreamType::Present, PhysicalLevelTechnique::None),
            num_values: 0,
            byte_length: 0,
            morton: None,
            rle: None,
        };

        let decoded = decode_physical_level_technique(&mut tile, &metadata).unwrap();
        assert_eq!(decoded, Vec::<u32>::new());
    }

    #[test]
    fn test_single_value_signed() {
        let bytes: Bytes = Bytes::from(vec![0x05, 0x00, 0x00, 0x00]);

        let mut tile: TrackedBytes = bytes.into();
        let metadata = StreamMetadata {
            logical: Logical::new(
                Some(LogicalStreamType::Dictionary(None)),
                LogicalLevelTechnique::None,
                LogicalLevelTechnique::None,
            ),
            physical: Physical::new(PhysicalStreamType::Present, PhysicalLevelTechnique::None),
            num_values: 1,
            byte_length: 4,
            morton: None,
            rle: None,
        };

        let result = decode_const_int_stream(&mut tile, &metadata, true).unwrap();

        assert_eq!(result, -3);
    }

    #[test]
    fn test_single_value_unsigned() {
        let bytes: Bytes = Bytes::from(vec![0x05, 0x00, 0x00, 0x00]);

        let mut tile: TrackedBytes = bytes.into();
        let metadata = StreamMetadata {
            logical: Logical::new(
                Some(LogicalStreamType::Dictionary(None)),
                LogicalLevelTechnique::None,
                LogicalLevelTechnique::None,
            ),
            physical: Physical::new(PhysicalStreamType::Present, PhysicalLevelTechnique::None),
            num_values: 1,
            byte_length: 4,
            morton: None,
            rle: None,
        };

        let result = decode_const_int_stream(&mut tile, &metadata, false).unwrap();

        assert_eq!(result, 5);
    }

    #[test]
    fn test_multiple_values_rle_case() {
        let bytes: Bytes = Bytes::from(vec![0x03, 0x00, 0x00, 0x00, 0x07, 0x00, 0x00, 0x00]);

        let mut tile: TrackedBytes = bytes.into();
        let metadata = StreamMetadata {
            logical: Logical::new(
                Some(LogicalStreamType::Dictionary(None)),
                LogicalLevelTechnique::None,
                LogicalLevelTechnique::None,
            ),
            physical: Physical::new(PhysicalStreamType::Present, PhysicalLevelTechnique::None),
            num_values: 2,
            byte_length: 8,
            morton: None,
            rle: None,
        };

        let result = decode_const_int_stream(&mut tile, &metadata, true).unwrap();

        assert_eq!(result, -4);
    }
}
