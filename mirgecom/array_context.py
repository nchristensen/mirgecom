import numpy as np
import loopy as lp
import grudge.loopy_dg_kernels as dgk
import hjson
from grudge.grudge_array_context import AutotuningArrayContext, unique_program_id, set_memory_layout
from grudge.loopy_dg_kernels.run_tests import exhaustive_search_v2, generic_test, convert
from pytools import memoize_method

class MirgecomAutotuningArrayContext(AutotuningArrayContext):

    #@memoize_method
    def get_generators(self, program):
        if program.default_entrypoint.name == "smooth_comp":
            tlist_generator = smooth_comp_tlist_generator
            from grudge.loopy_dg_kernels.generators import gen_autotune_list as pspace_generator
        else:
            tlist_generator, pspace_generator = super().get_generators(program)
        return tlist_generator, pspace_generator

    @memoize_method
    def transform_loopy_program(self, program):

        # Really just need to add metadata to the hjson file
        # Could convert the kernel itself to base 64 and store it
        # in the hjson file
        # TODO: Dynamically determine device id,
        #device_id = "NVIDIA Titan V"

        # These are the most compute intensive kernels
        to_optimize = {} #{"smooth_comp"}
        if program.default_entrypoint.name in to_optimize:
            print(program)
            for arg in program.default_entrypoint.args:
                print(arg.tags)
            exit()

        # Meshmode and Grudge kernels to autotune
        autotuned_kernels = {"smooth_comp"}

        if program.default_entrypoint.name in autotuned_kernels:
            print(program.default_entrypoint.name)
            print(program)

            # Set no_numpy and return_dict options here?
            program = lp.set_options(program, lp.Options(no_numpy=True, return_dict=True))
            program = set_memory_layout(program)
            pid = unique_program_id(program)
            print(pid)
            hjson_file_str = f"hjson/{program.default_entrypoint.name}_{pid}.hjson"

            try:
                # Attempt to read from a transformation file in the current directory first,
                # then try to read from the package files - this is not currently implemented
                # Maybe should have ability to search in arbitrary specified directories.

                hjson_file = open(hjson_file_str, "rt")
                transformations = dgk.load_transformations_from_file(hjson_file,
                    ["transformations"])
                hjson_file.close()
                print("LOCATED TRANSFORMATION:", hjson_file_str)

            except FileNotFoundError as e:
                
                search_fn = exhaustive_search_v2#random_search
                tlist_generator, pspace_generator = self.get_generators(program)
                transformations = self.autotune_and_save(self.queue, program, search_fn,
                                      tlist_generator, pspace_generator, hjson_file_str)

                """
                # Maybe the generators should be classes so we can use inheritance.
                if program.default_entrypoint.name == "smooth_comp":
                    tlist_generator = smooth_comp_tlist_generator                    
                    from grudge.loopy_dg_kernels.generators import gen_autotune_list as pspace_generator
                avg_time, transformations, data = search_fn(self.queue, program, generic_test, 
                                                pspace_generator, tlist_generator, time_limit=np.inf)

                od = {"transformations": transformations}
                out_file = open(hjson_file_str, "wt+")
                hjson.dump(od, out_file,default=convert)
                out_file.close()
                """

            program = dgk.apply_transformation_list(program, transformations)
        else:
            program = super().transform_loopy_program(program)

        return program

#for iel, idof
#    result[iel, idof] = reduce(sum, [kdof], vec[iel, kdof]*vec[iel, kdof]*modes_active_flag[kdof]) / reduce(sum, [jdof], vec[iel, jdof]*vec[iel, jdof] + (1e-12) / ndiscr_nodes_in)  {id=insn}
#end iel, idof
def smooth_comp_tlist_generator(params, **kwargs):
    trans_list = []
    kio, kii, iio, iii, ji = params
    knl = kwargs["knl"]

    trans_list.append(["split_iname", ["iel", kio], {"outer_tag": "g.0", "slabs":(0,1)}])
    trans_list.append(["split_iname", ["iel_inner", kii],
        {"outer_tag": "ilp", "inner_tag":"l.0", "slabs":(0,1)}])
    trans_list.append(["split_iname", ["idof", iio], {"outer_tag": "g.1", "slabs":(0,0)}])
    trans_list.append(["split_iname", ["idof_inner", iii],
        {"outer_tag": "ilp", "inner_tag":"l.1", "slabs":(0,1)}])
    # Should the i loop have (0,1) slabs for both?

    """
    for arg in knl.default_entrypoint.args:

        if "vec" == arg.name:
            trans_list.append(["add_prefetch", ["vec", "j,e_inner_outer,e_inner_inner"],
                {"temporary_name":"vecf", "default_tag":"l.auto"}])
            trans_list.append(["tag_array_axes", ["vecf", "f,f"]])
        elif "jac" == arg.name:
            trans_list.append(["add_prefetch", ["jac", "j,e_inner_outer,e_inner_inner"],
                {"temporary_name":"jacf", "default_tag":"l.auto"}])
            trans_list.append(["tag_array_axes", ["jacf", "f,f"]])
        elif "arg2" == arg.name and IsDOFArray() in arg.tags:
            trans_list.append(["add_prefetch", ["arg2", "j,e_inner_outer,e_inner_inner"],
                {"temporary_name":"arg2f", "default_tag":"l.auto"}])
            trans_list.append(["tag_array_axes", ["arg2f", "f,f"]])
        elif "arg1" == arg.name and IsDOFArray() in arg.tags:
            trans_list.append(["add_prefetch", ["arg1", "j,e_inner_outer,e_inner_inner"],
                {"temporary_name":"arg1f", "default_tag":"l.auto"}])
            trans_list.append(["tag_array_axes", ["arg1f", "f,f"]])
        elif "arg0" == arg.name and IsDOFArray() in arg.tags:
            trans_list.append(["add_prefetch",
                ["arg0", "i_inner_outer,i_inner_inner,e_inner_outer,e_inner_inner"],
                {"temporary_name":"arg0f", "default_tag":"l.auto"}])
            trans_list.append(["tag_array_axes", ["arg0f", "f,f"]])
    """

    trans_list.append(["split_iname", ["kdof", ji], {"outer_tag":"for", "inner_tag":"for"}])
    trans_list.append(["split_iname", ["jdof", ji], {"outer_tag":"for", "inner_tag":"for"}])
    trans_list.append(["add_inames_for_unused_hw_axes"]) 
    return trans_list 
