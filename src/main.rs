#![cfg_attr(not(debug_assertions), deny(warnings))]
#![doc(test(attr(deny(warnings))))]
#![doc(test(attr(warn(
                bare_trait_objects,
                clippy::cargo,
                clippy::pedantic,
                elided_lifetimes_in_paths,
                missing_copy_implementations,
                single_use_lifetimes,
                ))))]
#![warn(
    bare_trait_objects,
    clippy::cargo,
    clippy::pedantic,
    elided_lifetimes_in_paths,
    missing_copy_implementations,
    single_use_lifetimes,
    unused_extern_crates
)]
#![allow(clippy::multiple_crate_versions)]

use anyhow::Error as AnyhowError;

mod day1 {
    use {anyhow::{anyhow, Context, Error as AnyhowError}, std::num::NonZeroU64};

    const PART_ONE_INPUT: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/day_input/1p1"));

    fn calculate_single_fuel_round(mass: NonZeroU64) -> u64 {
        (mass.get() / 3).saturating_sub(2)
    }

    const PART_TWO_INPUT: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/day_input/1p2"));

    fn calculate_fuel_cost(mass: NonZeroU64) -> u64 {
        let mut mass = mass;
        let mut sum = 0;
        loop {
            match calculate_single_fuel_round(mass) {
                0 => break,
                units => {
                    sum += units;
                    mass = NonZeroU64::new(units).unwrap();
                }
            }
        }
        sum
    }

    fn for_input<F>(input: &str, f: F) -> Result<u64, AnyhowError>
        where
            F: Fn(NonZeroU64) -> u64,
        {
            input
                .lines()
                .try_fold(0u64, move |acc, line| {
                    acc.checked_add({
                        let parsed = line
                            .parse::<u64>()
                            .with_context(move || anyhow!("parse failed: {:?}", line))?;
                        let validated = NonZeroU64::new(parsed)
                            .ok_or_else(|| anyhow!("can't have entry with 0 as value"))?;
                        f(validated)
                    }).ok_or(anyhow!("overflow after add"))
                })
        }

    pub(crate) fn solutions() -> Result<(), AnyhowError> {
        println!("day 1");

        for_input(PART_ONE_INPUT, calculate_single_fuel_round)
            .map(|answer| println!("part 1: {}", answer))?;

        for_input(PART_TWO_INPUT, calculate_fuel_cost)
            .map(|answer| println!("part 2: {}", answer))
    }
}

mod day2 {
    use {
        anyhow::{anyhow, Context, Error as AnyhowError},
        lazy_format::lazy_format,
        std::{convert::TryFrom, fmt::Display},
    };

    const PART_ONE_INPUT: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/day_input/2p1"));

    fn run_intcode_machine(mut slots: &mut [u32]) -> Result<(), AnyhowError> {
        let mut program_counter = 0;

        loop {
            fn binary_op<F>(
                op_desc: &str,
                slots: &mut [u32],
                program_counter: usize,
                f: F,
                ) -> Result<usize, AnyhowError>
                where
                    F: Fn(u32, u32, &mut u32) -> Result<(), AnyhowError>,
                {
                    const NUM_ARGS: usize = 3;

                    ((|| {
                        let mut curr_arg_idx = 0;
                        macro_rules! next_arg {
                            ($method: ident) => {{
                                let idx = program_counter.checked_add(curr_arg_idx + 1).unwrap();
                                let arg = slots.get(idx)
                                    .ok_or_else(|| anyhow!(
                                        "insufficient number of slots for {} opcode (0x{:X})",
                                        op_desc,
                                        program_counter
                                    ))?;
                                let arg = usize::try_from(*arg)
                                    .with_context(|| anyhow!(
                                        "unable to convert address 0x{:X} to `usize`",
                                        arg,
                                    ))?;
                                let ret = slots
                                    .$method(arg)
                                    .ok_or_else(|| anyhow!(
                                        "address 0x{:X} specified at 0x{:X} is invalid",
                                        curr_arg_idx,
                                        program_counter,
                                    ))?;
                                curr_arg_idx += 1;
                                ret
                            }};
                        }

                        let mut get = || Result::<_, AnyhowError>::Ok(*next_arg!(get)) ;

                        let first = get()?;
                        let second = get()?;
                        let store_idx = next_arg!(get_mut);
                        debug_assert_eq!(NUM_ARGS, curr_arg_idx);
                        f(first, second, store_idx)
                    })())
                    .with_context(|| anyhow!(
                        "failed executing instruction at 0x{:X}",
                        program_counter
                    ))?;

                Ok(NUM_ARGS)
            }

            let opcode = slots
                .get(program_counter)
                .ok_or_else(|| anyhow!("invalid program counter {}", program_counter))?;

            let slots_read = 1 + match opcode {
                1 => binary_op("addition", &mut slots, program_counter, |first, second, result| {
                    *result = first
                        .checked_add(second)
                        .ok_or_else(|| anyhow!(
                            "overflow performing addition ({} + {})",
                            first,
                            second,
                        ))?;
                    Ok(())
                })?,
                2 => binary_op("multiplication", &mut slots, program_counter, |first, second, result| {
                    *result = first
                        .checked_mul(second)
                        .ok_or_else(|| anyhow!(
                            "overflow performing multiplication ({} * {})",
                            first,
                            second,
                        ))?;
                    Ok(())
                })?,
                99 => break Ok(()),
                _ => return Err(anyhow!("invalid opcode {} at 0x{:X}", opcode, program_counter))
            };

            program_counter = program_counter.checked_add(slots_read).unwrap();
        }
    }

    fn test(input: &[u32], expected_output: &[u32]) {
        let mut slots = Vec::from(input);
        run_intcode_machine(&mut slots).unwrap();
        assert_eq!(slots.as_slice(), expected_output);
    }

    fn parse_slots(s: &str) -> Result<Vec<u32>, AnyhowError> {
        s.split(',')
            .map(|s| s.trim().parse().with_context(|| anyhow!("slot parse failed for {:?}", s)))
            .collect::<Result<Vec<u32>, _>>()
    }

    fn part1() -> Result<u32, AnyhowError> {
        test(&[1, 0, 0, 0, 99], &[2, 0, 0, 0, 99]);
        test(&[2, 3, 0, 3, 99], &[2, 3, 0, 6, 99]);
        test(&[2, 4, 4, 5, 99, 0], &[2, 4, 4, 5, 99, 9801]);
        test(&[1, 1, 1, 4, 99, 5, 6, 0, 99], &[30, 1, 1, 4, 2, 5, 6, 0, 99]);

        let mut slots = parse_slots(PART_ONE_INPUT)?;

        slots[1] = 12;
        slots[2] = 2;
        run_intcode_machine(slots.as_mut_slice())?;
        Ok(slots[0])
    }

    const PART_TWO_INPUT: &str = PART_ONE_INPUT;

    fn part2() -> Result<impl Display, AnyhowError> {
        let slots = parse_slots(PART_TWO_INPUT)?;

        let mut solution = None;

        'outer: for noun in 1..=99 {
            for verb in 1..=99 {
                let mut slots = slots.clone();
                slots[1] = noun;
                slots[2] = verb;
                run_intcode_machine(slots.as_mut_slice())?;
                if slots[0] == 19690720 {
                    solution = Some((noun, verb));
                    break 'outer;
                }
            }
        }

        solution.map(|(noun, verb)| lazy_format!(
            "100 * {} + {} = {}",
            noun,
            verb,
            noun.checked_mul(100).unwrap().checked_add(verb).unwrap()
        )).ok_or_else(|| anyhow!("solution not found"))
    }

    pub(crate) fn solutions() -> Result<(), AnyhowError> {
        println!("day 2");
        println!("  part 1: {}", part1()?);
        println!("  part 2: {}", part2()?);
        Ok(())
    }
}

mod day3 {
    use {
        anyhow::{anyhow, ensure, Context, Error as AnyhowError},
        lazy_format::lazy_format,
        std::{cmp::{min, max}, convert::TryFrom},
    };

    const PART_ONE_INPUT: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/day_input/3p1"));

    fn find_closest_manhattan_distance_of_crossings(input: &str) -> Result<usize, AnyhowError> {
        #[derive(Clone, Copy, Debug, Eq, PartialEq)]
        enum Direction {
            Up,
            Down,
            Left,
            Right,
        }

        let mut lines_iter = input.lines().map(|l| {
            l.split(',').map(|raw_instruction| {
                use Direction::*;
                let mut chars = raw_instruction.chars();

                let direction = match chars.next().ok_or_else(|| anyhow!("movement string is empty"))? {
                    'U' => Up,
                    'D' => Down,
                    'L' => Left,
                    'R' => Right,
                    c => return Err(anyhow!("unrecognized direction char {:?}", c)),
                };

                let num_moves = match chars.clone().next() {
                    Some(c) if c.is_digit(10) => Ok(()),
                    c => Err(anyhow!("expected first character of number of moves to be a digit; got {:?}", c)),
                }.and_then(|()| {
                    let s = chars.as_str();
                    s.parse::<isize>().map_err(AnyhowError::new)
                }).with_context(|| anyhow!("failed to parse number of moves"))?;

                Ok((direction, num_moves))
            }).collect::<Result<Vec<_>, _>>()
        });

        let line_err = || anyhow!("expected 2 lines of input");
        let first_wire_movements = lines_iter.next().ok_or_else(line_err)??;
        let second_wire_movements = lines_iter.next().ok_or_else(line_err)??;
        ensure!(lines_iter.next().is_none(), line_err());

        println!("first_wire_movements: {:?}", second_wire_movements);
        println!("second_wire_movements: {:?}", second_wire_movements);

        // first pass: just see how far everything goes
        let origin_x = 0isize;
        let origin_y = 0isize;

        let mut curr_x = origin_x;
        let mut curr_y = origin_y;

        let mut min_x = curr_x;
        let mut max_x = curr_x;
        let mut min_y = curr_y;
        let mut max_y = curr_y;

        [&first_wire_movements, &second_wire_movements].iter().for_each(|path| {
            path.iter().for_each(|(direction, num_moves)| {
                use Direction::*;

                let (coord, min_, max_, is_positive) = match direction {
                    Up => (&mut curr_y, &mut min_y, &mut max_y, true),
                    Down => (&mut curr_y, &mut min_y, &mut max_y, false),
                    Left => (&mut curr_x, &mut min_x, &mut max_x, false),
                    Right => (&mut curr_x, &mut min_x, &mut max_x, true),
                };

                let multiplier = if is_positive { 1 } else { -1 };

                *coord = coord.checked_add(num_moves.checked_mul(multiplier).unwrap()).unwrap();
                *min_ = min(*min_, *coord);
                *max_ = max(*max_, *coord);
            })
        });

        let size_x = usize::try_from(max_x.checked_sub(min_x).unwrap().checked_add(1).unwrap()).unwrap();
        let size_y = usize::try_from(max_y.checked_sub(min_y).unwrap().checked_add(1).unwrap()).unwrap();

        println!(
            "grid needs to span from {:?} to {:?} (so the size would be {:?})",
            (min_x, min_y),
            (max_x, max_y),
            (size_x, size_y),
        );

        let mut grid = vec![(false, false); size_x.checked_mul(size_y).unwrap()];

        let origin_adjusted_x = usize::try_from(origin_x.checked_sub(min_x).unwrap()).unwrap();
        let origin_adjusted_y = usize::try_from(origin_y.checked_sub(min_y).unwrap()).unwrap();

        let mut closest_crossing_point = None;

        let path_and_recorder_pairs: [(&Vec<(Direction, isize)>, &dyn Fn(&mut (bool, bool))); 2] = [
            (&first_wire_movements, &|(first, _second): &mut (_, _)| *first = true),
            (&second_wire_movements, &|(_first, second): &mut (_, _)| *second = true),
        ];

        path_and_recorder_pairs.iter().for_each(|(path, path_recorder)| {
            println!("NEW WIRE");

            let mut curr_x = origin_adjusted_x;
            let mut curr_y = origin_adjusted_y;

            path.iter().copied().for_each(|(direction, num_moves)| {
                use Direction::*;

                println!("Executing move {:?}", (direction, num_moves));
                let (mut_coord, other_coord, is_positive, is_vertical) = match direction {
                    Up => (&mut curr_y, curr_x, true, true),
                    Down => (&mut curr_y, curr_x, false, true),
                    Left => (&mut curr_x, curr_y, false, false),
                    Right => (&mut curr_x, curr_y, true, false),
                };

                let step: &dyn Fn(usize) -> Option<usize> = if is_positive {
                    &|x| x.checked_add(1)
                } else {
                    &|x| x.checked_sub(1)
                };

                (0..num_moves).for_each(|_i| {
                    *mut_coord = step(*mut_coord).unwrap();

                    let (curr_x, curr_y) = if is_vertical {
                        (other_coord, *mut_coord)
                    } else {
                        (*mut_coord, other_coord)
                    };

                    println!("Moving to {:?}", (curr_x, curr_y));

                    let idx = curr_x.checked_add(curr_y.checked_mul(size_x).unwrap()).unwrap();
                    let record = match grid.get_mut(idx) {
                        Some(r) => r,
                        None => panic!(
                            "grid size is {}, but tried accessing {:?} (idx {})",
                            grid.len(),
                            (curr_x, curr_y),
                            idx,
                        ),
                    };

                    path_recorder(record);

                    if let (true, true) = record {
                        println!("Crossing: {:?}", (
                            isize::try_from(curr_x).unwrap() - isize::try_from(origin_adjusted_x).unwrap(),
                            isize::try_from(curr_y).unwrap() - isize::try_from(origin_adjusted_y).unwrap(),
                        ));

                        let abs_diff = |a, b| if a < b {
                            b - a
                        } else {
                            a - b
                        };

                        let manhattan_distance = if is_vertical {
                            abs_diff(origin_adjusted_y, *mut_coord) + abs_diff(origin_adjusted_x, other_coord)
                        } else {
                            abs_diff(origin_adjusted_x, *mut_coord) + abs_diff(origin_adjusted_y, other_coord)
                        };

                        closest_crossing_point = closest_crossing_point
                            .map(|d| min(d, manhattan_distance))
                            .or(Some(manhattan_distance));
                    }
                })
            })
        });

        closest_crossing_point.ok_or_else(|| anyhow!("no crossing points found"))
    }

    fn part1() -> Result<usize, AnyhowError> {
        assert_eq!(
            find_closest_manhattan_distance_of_crossings(
                "R8,U5,L5,D3\nU7,R6,D4,L4"
            ).unwrap(),
            6,
        );

        assert_eq!(
            find_closest_manhattan_distance_of_crossings(
                "R75,D30,R83,U83,L12,D49,R71,U7,L72\nU62,R66,U55,R34,D71,R55,D58,R83"
            ).unwrap(),
            159,
        );

        assert_eq!(
            find_closest_manhattan_distance_of_crossings(
                "R98,U47,R26,D63,R33,U87,L62,D20,R33,U53,R51\nU98,R91,D20,R16,D67,R40,U7,R15,U6,R7"
            ).unwrap(),
            135,
        );

        find_closest_manhattan_distance_of_crossings(PART_ONE_INPUT)
    }

    pub(crate) fn solutions() -> Result<(), AnyhowError> {
        println!("day 3");
        println!("  part 1: {}", {
            let part1 = part1()?;
            lazy_format!("{:?}", part1)
        });
        Ok(())
    }
}

fn main() -> Result<(), AnyhowError> {
    day1::solutions()?;
    day2::solutions()?;
    day3::solutions()?;
    Ok(())
}
