/* QEMU mps2-an386 (Cortex-M4). Code region at 0x0, SSRAM at 0x20000000. */
MEMORY
{
  FLASH : ORIGIN = 0x00000000, LENGTH = 4M
  RAM   : ORIGIN = 0x20000000, LENGTH = 4M
}

_stack_start = ORIGIN(RAM) + LENGTH(RAM);
