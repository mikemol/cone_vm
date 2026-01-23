from typing import TYPE_CHECKING, assert_type

import jax.numpy as jnp

import prism_vm as pv


if TYPE_CHECKING:
    vm = pv.PrismVM()
    bsp = pv.PrismVM_BSP()
    legacy = pv.PrismVM_BSP_Legacy()

    mptr = vm.parse(["zero"])
    lid = bsp.parse(["zero"])
    aptr = legacy.parse(["zero"])
    assert_type(mptr, pv.ManifestPtr)
    assert_type(lid, pv.LedgerId)
    assert_type(aptr, pv.ArenaPtr)

    assert_type(vm.eval(mptr), pv.ManifestPtr)
    assert_type(bsp.decode(lid), str)

    h_count = pv._active_prefix_count(pv.init_arena())
    assert_type(h_count, pv.HostInt)
    h_corrupt = pv.ledger_has_corrupt(pv.init_ledger())
    assert_type(h_corrupt, pv.HostBool)

    prov = pv._provisional_ids(jnp.array([0], dtype=jnp.int32))
    committed = pv._identity_q(prov)
    assert_type(committed, pv.CommittedIds)
    _ = pv._identity_q(committed)  # type: ignore

    frontier = pv._committed_ids(jnp.array([0], dtype=jnp.int32))
    led = pv.init_ledger()
    led2, prov_frontier, _, q_map = pv.cycle_candidates(led, frontier)
    assert_type(prov_frontier, pv.ProvisionalIds)
    assert_type(q_map, pv.QMap)
    committed_frontier = pv.apply_q(q_map, prov_frontier)
    assert_type(committed_frontier, pv.CommittedIds)

    bad_manifest: pv.ManifestPtr = lid  # type: ignore
    bad_ledger: pv.LedgerId = mptr  # type: ignore
    _ = vm.eval(lid)  # type: ignore
    _ = bsp.decode(mptr)  # type: ignore
    _ = legacy.decode(mptr)  # type: ignore
