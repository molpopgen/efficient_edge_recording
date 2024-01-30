// This is a rust implementation of the example
// found in tskit-c

use std::num::NonZeroUsize;

use anyhow::Result;
use clap::Parser;
use rand::prelude::*;
use rand::SeedableRng;

#[derive(Default)]
struct BufferedEdges {
    left: Vec<tskit::Position>,
    right: Vec<tskit::Position>,
    child: Vec<tskit::NodeId>,
    next: Vec<Option<NonZeroUsize>>,
}

impl BufferedEdges {
    fn len(&self) -> usize {
        debug_assert_eq!(self.left.len(), self.right.len());
        debug_assert_eq!(self.left.len(), self.child.len());
        debug_assert_eq!(self.left.len(), self.next.len());
        self.left.len()
    }

    fn clear(&mut self) {
        self.left.clear();
        self.right.clear();
        self.child.clear();
        self.next.clear();
    }
}

#[derive(Default)]
struct EdgeBuffer {
    first: Vec<Option<usize>>,
    last: Vec<Option<usize>>,
    edges: BufferedEdges,
    num_nodes_before_births: usize,
}

impl EdgeBuffer {
    fn init(&mut self, current_num_nodes: usize) {
        // FIXME: this fn should take the tables as an arg to
        // make use less error prone
        self.first.clear();
        self.last.clear();
        self.first.resize(current_num_nodes, None);
        self.last.resize(current_num_nodes, None);
        self.edges.clear();
        self.num_nodes_before_births = current_num_nodes;
    }

    fn add_node(&mut self) {
        self.first.push(None);
        self.last.push(None);
    }

    fn add_edge<L, R>(&mut self, left: L, right: R, parent: tskit::NodeId, child: tskit::NodeId)
    where
        L: Into<tskit::Position>,
        R: Into<tskit::Position>,
    {
        let pindex = usize::try_from(parent).unwrap();
        let current_num_edges = self.edges.len();
        match self.first[pindex] {
            Some(_) => {
                assert!(current_num_edges > 0);
                let last = self.last[pindex].unwrap();
                self.edges.left.push(left.into());
                self.edges.right.push(right.into());
                self.edges.child.push(child);
                self.edges.next[last] = Some(NonZeroUsize::try_from(current_num_edges).unwrap());
                self.edges.next.push(None);
                self.last[pindex] = Some(current_num_edges);
            }
            None => {
                assert!(self.last[pindex].is_none());
                self.edges.left.push(left.into());
                self.edges.right.push(right.into());
                self.edges.child.push(child);
                self.edges.next.push(None);
                self.first[pindex] = Some(current_num_edges);
                self.last[pindex] = Some(current_num_edges);
            }
        }
    }
}

fn rotate_edges(tables: &mut tskit::TableCollection, mid: usize) {
    let num_edges = tables.edges().num_rows().as_usize();
    let left =
        unsafe { std::slice::from_raw_parts_mut((*tables.as_mut_ptr()).edges.left, num_edges) };
    let right =
        unsafe { std::slice::from_raw_parts_mut((*tables.as_mut_ptr()).edges.right, num_edges) };
    let parent =
        unsafe { std::slice::from_raw_parts_mut((*tables.as_mut_ptr()).edges.parent, num_edges) };
    let child =
        unsafe { std::slice::from_raw_parts_mut((*tables.as_mut_ptr()).edges.child, num_edges) };
    left.rotate_left(mid);
    right.rotate_left(mid);
    parent.rotate_left(mid);
    child.rotate_left(mid);
}

fn simulate(
    seed: u64,
    popsize: usize,
    num_generations: i32,
    simplify_interval: i32,
) -> Result<tskit::TreeSequence> {
    if popsize == 0 {
        return Err(anyhow::Error::msg("popsize must be > 0"));
    }
    if num_generations == 0 {
        return Err(anyhow::Error::msg("num_generations must be > 0"));
    }
    if simplify_interval == 0 {
        return Err(anyhow::Error::msg("simplify_interval must be > 0"));
    }
    let mut tables = tskit::TableCollection::new(1.0)?;

    // create parental nodes
    let mut parents_and_children = {
        let mut temp = vec![];
        let parental_time = f64::from(num_generations);
        for _ in 0..popsize {
            let node = tables.add_node(0, parental_time, -1, -1)?;
            temp.push(node);
        }
        temp
    };

    let mut buffer = EdgeBuffer::default();
    buffer.init(tables.nodes().num_rows().try_into().unwrap());

    // allocate space for offspring nodes
    parents_and_children.resize(2 * parents_and_children.len(), tskit::NodeId::NULL);

    // Construct non-overlapping mutable slices into our vector.
    let (mut parents, mut children) = parents_and_children.split_at_mut(popsize);

    let parent_picker = rand::distributions::Uniform::new(0, popsize);
    let breakpoint_generator = rand::distributions::Uniform::new(0.0, 1.0);
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    for birth_time in (0..num_generations).rev() {
        for c in children.iter_mut() {
            let bt = f64::from(birth_time);
            let child = tables.add_node(0, bt, -1, -1)?;
            buffer.add_node();
            let left_parent = parents
                .get(parent_picker.sample(&mut rng))
                .ok_or_else(|| anyhow::Error::msg("invalid left_parent index"))?;
            let right_parent = parents
                .get(parent_picker.sample(&mut rng))
                .ok_or_else(|| anyhow::Error::msg("invalid right_parent index"))?;
            let breakpoint = breakpoint_generator.sample(&mut rng);
            buffer.add_edge(0., breakpoint, *left_parent, child);
            buffer.add_edge(breakpoint, 1.0, *right_parent, child);
            *c = child;
        }

        if birth_time % simplify_interval == 0 {
            let num_edges = tables.edges().num_rows().as_usize();
            for (parent, first) in buffer
                .first
                .iter()
                .enumerate()
                .skip(buffer.num_nodes_before_births)
                .rev()
            {
                let parent_as_node = tskit::NodeId::from(i32::try_from(parent).unwrap());
                let mut edge_index = *first;
                while let Some(index) = edge_index {
                    tables
                        .add_edge(
                            buffer.edges.left[index],
                            buffer.edges.right[index],
                            parent_as_node,
                            buffer.edges.child[index],
                        )
                        .unwrap();
                    edge_index = buffer.edges.next[index].map(usize::from);
                }
            }
            for (parent, first) in buffer
                .first
                .iter()
                .enumerate()
                .take(buffer.num_nodes_before_births)
            {
                let parent_as_node = tskit::NodeId::from(i32::try_from(parent).unwrap());
                let mut edge_index = *first;
                while let Some(index) = edge_index {
                    tables
                        .add_edge(
                            buffer.edges.left[index],
                            buffer.edges.right[index],
                            parent_as_node,
                            buffer.edges.child[index],
                        )
                        .unwrap();
                    edge_index = buffer.edges.next[index].map(usize::from);
                }
            }
            rotate_edges(&mut tables, num_edges);
            tables.check_integrity(tskit::TableIntegrityCheckFlags::CHECK_EDGE_ORDERING)?;
            if let Some(idmap) =
                tables.simplify(children, tskit::SimplificationOptions::default(), true)?
            {
                // remap child nodes
                for o in children.iter_mut() {
                    *o = idmap[usize::try_from(*o)?];
                }
            }
            buffer.init(tables.nodes().num_rows().as_usize());
        }
        std::mem::swap(&mut parents, &mut children);
    }

    tables.build_index()?;
    let treeseq = tables.tree_sequence(tskit::TreeSequenceFlags::default())?;

    Ok(treeseq)
}

#[derive(Clone, clap::Parser)]
struct SimParams {
    seed: u64,
    popsize: usize,
    num_generations: i32,
    simplify_interval: i32,
    treefile: Option<String>,
}

fn main() -> Result<()> {
    let params = SimParams::parse();
    let treeseq = simulate(
        params.seed,
        params.popsize,
        params.num_generations,
        params.simplify_interval,
    )?;

    if let Some(treefile) = &params.treefile {
        treeseq.dump(treefile, 0)?;
    }

    Ok(())
}

#[test]
fn simple_buffer_test() {
    let mut buffer = EdgeBuffer::default();
    let mut tables = tskit::TableCollection::new(10.).unwrap();
    buffer.init(tables.nodes().num_rows().as_usize());
    tables.add_node(0, 10., -1, -1).unwrap();
    buffer.add_node();
    tables.add_node(0, 9., -1, -1).unwrap();
    buffer.add_node();
    tables.add_node(0, 9., -1, -1).unwrap();
    buffer.add_node();

    buffer.add_edge(0., 5., 0.into(), 1.into());
    buffer.add_edge(0., 5., 0.into(), 2.into());

    assert_eq!(buffer.first[0], Some(0));
    assert_eq!(buffer.last[0], Some(1));
    assert_eq!(
        buffer.edges.next[0],
        Some(NonZeroUsize::try_from(1).unwrap())
    );

    let mut output = vec![];
    for (parent, first) in buffer.first.iter().enumerate() {
        let mut edge_index = *first;
        while let Some(index) = edge_index {
            output.push(index);
            assert_eq!(parent, 0);
            edge_index = buffer.edges.next[index].map(usize::from);
        }
    }
    assert_eq!(output, [0, 1]);
}
