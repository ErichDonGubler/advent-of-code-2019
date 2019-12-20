use {
    anyhow::{anyhow, Context, Error as AnyhowError},
    std::{convert::TryFrom, str::FromStr},
};

#[derive(Clone, Debug)]
pub struct IntcodeMachineState {
    pub slots: Vec<u32>,
    pub program_counter: usize,
}

impl FromStr for IntcodeMachineState {
    type Err = AnyhowError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(Self {
            slots: s
                .split(',')
                .map(|s| {
                    s.trim()
                        .parse()
                        .with_context(|| anyhow!("slot parse failed for {:?}", s))
                })
                .collect::<Result<Vec<u32>, _>>()?,
            program_counter: 0,
        })
    }
}

impl IntcodeMachineState {
    pub fn new(slots: Vec<u32>) -> Self {
        Self {
            slots,
            program_counter: 0,
        }
    }
}

pub trait IntcodeEngine {
    fn execute(&self, state: &mut IntcodeMachineState) -> Result<(), AnyhowError>;
}

pub(crate) fn binary_op<F>(
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
                let arg = slots.get(idx).ok_or_else(|| {
                    anyhow!(
                        "insufficient number of slots for {} opcode (0x{:X})",
                        op_desc,
                        program_counter
                    )
                })?;
                let arg = usize::try_from(*arg).with_context(|| {
                    anyhow!("unable to convert address 0x{:X} to `usize`", arg,)
                })?;
                let ret = slots.$method(arg).ok_or_else(|| {
                    anyhow!(
                        "address 0x{:X} specified at 0x{:X} is invalid",
                        curr_arg_idx,
                        program_counter,
                    )
                })?;
                curr_arg_idx += 1;
                ret
            }};
        }

        let mut get = || Result::<_, AnyhowError>::Ok(*next_arg!(get));

        let first = get()?;
        let second = get()?;
        let store_idx = next_arg!(get_mut);
        debug_assert_eq!(NUM_ARGS, curr_arg_idx);
        f(first, second, store_idx)
    })())
    .with_context(|| anyhow!("failed executing instruction at 0x{:X}", program_counter))?;

    Ok(NUM_ARGS)
}
