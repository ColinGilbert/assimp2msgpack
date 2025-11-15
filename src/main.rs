use msgpacker::prelude::*;
use noobwerkz::serialized_model::*;
use russimp_ng::{Vector3D, material::*};
use russimp_ng::{node::Node, scene::*};
use std::borrow::*;
use std::cell::*;
use std::collections::*;
use std::fs::*;
use std::io::*;
use std::rc::*;

// The function gathers the bone names and inverse bind pose matrices in depth-first order,
fn recursive_helper(
    node: &Node,
    bones_to_inverse_bind_poses: &HashMap<String, [[f32; 4]; 4]>,
    bone_names_vec: &mut Vec<String>,
    inverse_bind_matrices_vec: &mut Vec<[[f32; 4]; 4]>,
) {
    let name = &node.name;
    if bones_to_inverse_bind_poses.contains_key(name) && !bone_names_vec.contains(name) {
        bone_names_vec.push(name.clone());
        inverse_bind_matrices_vec.push(bones_to_inverse_bind_poses[name]);
    }

    for c in node.children.borrow().iter() {
        let child: &Node = c.borrow();
        recursive_helper(
            child,
            bones_to_inverse_bind_poses,
            bone_names_vec,
            inverse_bind_matrices_vec,
        );
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        println!("Usage: assimp2msgpack <FILENAME>");
        return;
    }

    let filename = args[1].clone();

    let post_process_flags = vec![
        PostProcess::CalculateTangentSpace,
        PostProcess::Triangulate,
        PostProcess::JoinIdenticalVertices,
        PostProcess::SortByPrimitiveType,
        PostProcess::GenerateNormals, // Example: Generate normals if missing
        PostProcess::GenerateUVCoords, // Example: Generate UVs if missing
    ];
    let mut bones_to_inverse_bind_poses = HashMap::<String, [[f32; 4]; 4]>::new();
    let mut result = SerializedModel::new();
    let mut has_bones = false;
    let scene = Scene::from_file(&filename, post_process_flags).unwrap();
    for m in scene.meshes {
        let mut mesh = SerializedMesh::new();
        for v in m.vertices {
            mesh.positions.push([v.x, v.y, v.z]);
        }
        for n in m.normals {
            mesh.normals.push([n.x, n.y, n.z]);
        }

        for t in m.texture_coords {
            for tex in t.unwrap_or_else(|| Vec::<Vector3D>::new()) {
                mesh.uvs.push([tex.x, tex.y]);
            }
        }

        if m.bones.len() > 0 {
            has_bones = true;
            mesh.bone_indices.resize(mesh.positions.len(), [0, 0, 0, 0]);
            mesh.bone_weights
                .resize(mesh.positions.len(), [0.0, 0.0, 0.0, 0.0]);
            for b in m.bones {
                let name = b.name.clone();
                mesh.bone_names.push(name.clone());
                let bone_name_idx = mesh.bone_names.len() - 1;
                for w in b.weights {
                    let id = w.vertex_id;
                    let weight = w.weight;
                    let mut i = 0;
                    while i < mesh.bone_weights[id as usize].len() {
                        let w = mesh.bone_weights[id as usize][i];
                        if w == 0.0 {
                            mesh.bone_weights[id as usize][i] = weight;
                            mesh.bone_indices[id as usize][i] = bone_name_idx as u32;
                        }
                        i += 1;
                    }
                }
                let mut ibp = [[0.0 as f32; 4]; 4];
                // ibp[0][0] = b.offset_matrix.a1;
                // ibp[0][1] = b.offset_matrix.a2;
                // ibp[0][2] = b.offset_matrix.a3;
                // ibp[0][3] = b.offset_matrix.a4;
                // ibp[1][0] = b.offset_matrix.b1;
                // ibp[1][1] = b.offset_matrix.b2;
                // ibp[1][2] = b.offset_matrix.b3;
                // ibp[1][3] = b.offset_matrix.b4;
                // ibp[2][0] = b.offset_matrix.c1;
                // ibp[2][1] = b.offset_matrix.c2;
                // ibp[2][2] = b.offset_matrix.c3;
                // ibp[2][3] = b.offset_matrix.c4;
                // ibp[3][0] = b.offset_matrix.d1;
                // ibp[3][1] = b.offset_matrix.d2;
                // ibp[3][2] = b.offset_matrix.d3;
                // ibp[3][3] = b.offset_matrix.d4;

                ibp[0][0] = b.offset_matrix.a1;
                ibp[1][0] = b.offset_matrix.a2;
                ibp[2][0] = b.offset_matrix.a3;
                ibp[3][0] = b.offset_matrix.a4;
                
                ibp[0][1] = b.offset_matrix.b1;
                ibp[1][1] = b.offset_matrix.b2;
                ibp[2][1] = b.offset_matrix.b3;
                ibp[3][1] = b.offset_matrix.b4;
                
                ibp[0][2] = b.offset_matrix.c1;
                ibp[1][2] = b.offset_matrix.c2;
                ibp[2][2] = b.offset_matrix.c3;
                ibp[3][2] = b.offset_matrix.c4;
                
                ibp[0][3] = b.offset_matrix.d1;
                ibp[1][3] = b.offset_matrix.d2;
                ibp[2][3] = b.offset_matrix.d3;
                ibp[3][3] = b.offset_matrix.d4;


                bones_to_inverse_bind_poses.insert(name.clone(), ibp);
            }
        }

        for f in m.faces {
            for i in f.0 {
                mesh.indices.push(i);
            }
        }

        mesh.material_index = m.material_index;
        result.meshes.push(mesh);
    }

    if has_bones {
        // We use our recursive helper to gather all the bones and their offset matrices
        recursive_helper(
            scene.root.unwrap().borrow(),
            &bones_to_inverse_bind_poses,
            &mut result.bone_names,
            &mut result.inverse_bind_matrices,
        );

        // We now need to change the bone indices on all the serialized meshes to reflect the bones
        for mesh in result.meshes.iter_mut() {
            for bi in mesh.bone_indices.iter_mut() {
                let mut i = 0;
                while i < bi.len() {
                    let bone_name = &mesh.bone_names[bi[i] as usize];
                    let bone_pos = result
                        .bone_names
                        .iter()
                        .position(|name| name == bone_name)
                        .unwrap();
                    bi[i] = bone_pos as u32;
                    i += 1;
                }
            }
        }
    }

    for mat in scene.materials {
        let mut material = SerializedMaterial::new();

        if mat.textures.contains_key(&TextureType::Diffuse) {
            let diffuse_texture = &mat.textures[&TextureType::Diffuse];
            let diffuse =
                Rc::<RefCell<russimp_ng::material::Texture>>::try_unwrap(diffuse_texture.clone())
                    .unwrap()
                    .into_inner();
            material.diffuse_texture_path = diffuse.filename;
        }

        if mat.textures.contains_key(&TextureType::Normals) {
            let normals_texture = &mat.textures[&TextureType::Normals];
            let normals =
                Rc::<RefCell<russimp_ng::material::Texture>>::try_unwrap(normals_texture.clone())
                    .unwrap()
                    .into_inner();
            material.normals_texture_path = normals.filename;
        }

        if mat.textures.contains_key(&TextureType::Specular) {
            let specular_texture = &mat.textures[&TextureType::Specular];
            let specular =
                Rc::<RefCell<russimp_ng::material::Texture>>::try_unwrap(specular_texture.clone())
                    .unwrap()
                    .into_inner();

            material.specular_texture_path = specular.filename;
        }

        result.materials.push(material);
    }

    let mut buf = Vec::new();
    let _num_bytes = result.pack(&mut buf);

    let mut file = File::create("model.bin").unwrap();
    let _ = file.write_all(&buf);
}
