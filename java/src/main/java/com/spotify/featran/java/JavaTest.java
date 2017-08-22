/*
 * Copyright 2017 Spotify AB.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package com.spotify.featran.java;

import com.spotify.featran.transformers.Binarizer;
import com.spotify.featran.transformers.Identity;
import com.spotify.featran.transformers.OneHotEncoder;

import java.util.*;
import java.util.stream.Collectors;

public class JavaTest {

  static class Record {
    private int i;
    private double d;
    private String s;

    Record(int i, double d, String s) {
      this.i = i;
      this.d = d;
      this.s = s;
    }
  }

  public static void main(String[] args) {
    List<Record> data = new ArrayList<>();
    for (int i = 1; i <= 10; i++) {
      data.add(new Record(i, i / 10.0, "s" + i % 3));
    }

    JFeatureSpec<Record> spec = JFeatureSpec.<Record>of()
        .required(r -> (double) r.i, Identity.apply("id"))
        .required(r -> r.d, Binarizer.apply("bin", 0.5))
        .optional(r -> Optional.of(r.s), OneHotEncoder.apply("one-hot1"))
        .optional(r -> Optional.ofNullable(r.s), "missing", OneHotEncoder.apply("one-hot2"));

    JFeatureExtractor<Record> fe1 = spec.extract(data);
    String settings = fe1.featureSettings();
    print(fe1);

    List<Record> half = data.stream()
        .filter(r -> r.i <= 5)
        .collect(Collectors.toList());
    JFeatureExtractor<Record> fe2 = spec.extractWithSettings(half, settings);
    print(fe2);
  }

  private static void print(JFeatureExtractor<Record> fe) {
    System.out.println(fe.featureSettings());
    System.out.println(fe.featureNames());
    for (double[] v : fe.featureValuesDouble()) {
      System.out.println(Arrays.stream(v).boxed().collect(Collectors.toList()));
    }
  }

}
