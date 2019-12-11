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

fn main() -> Result<(), AnyhowError> {
    day1::solutions()?;
    day2::solutions()?;
}
